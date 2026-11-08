# Runbook: Set Up Stage 1 on AWS

This runbook carries out ADR 009 (`docs/adr/009-pay-per-use-hosting.md`) in an AWS account, one step at a time, with the AWS CLI. Each step says why it exists, gives the commands, a read-only check, and how to undo it. Nothing here is automated: an operator runs each step and reads its output before the next.

> **Note:** Several commands print ARNs, which contain the account ID. The commands below use `--query` to print only the fields needed, or send output to `/dev/null`. That covers successful output only: an error message from the CLI or Docker can include an ARN or the registry host. Keep the account ID out of anything committed or shared.

> **Note:** The function URL is created last (step 9), after the kill switch exists (step 8), and the kill switch is the first thing tested once the URL exists (step 10). Until step 9 the function has no public endpoint.

## 0. Before starting

Run every step from the repository root, in one shell; steps 8 and 10 use paths relative to it. The steps use the AWS CLI (version 2; `aws login` needs 2.32.0 or later), `docker` with `buildx`, `jq`, `ffmpeg`, `zip`, `curl` and `uv`.

ADR 009 precondition 0: the account's Lambda concurrency quota must be at least 101, because reserving 1 needs the unreserved count minus 100.

```bash
AWS=~/.local/bin/aws
"$AWS" service-quotas get-service-quota --service-code lambda --quota-code L-B99A9384 \
  --region us-east-1 --query "Quota.Value"
```

Continue only if this prints 101 or more. The quota code comes from this account's `list-service-quotas` output for Lambda ("Concurrent executions").

Set these in the shell that runs every later step. `ACCOUNT_ID` is read once into a variable; the commands here do not print it.

```bash
export AWS_REGION=us-east-1
REGION=us-east-1
ACCOUNT_ID="$("$AWS" sts get-caller-identity --query Account --output text)"
FUNCTION=harmonyrestorer-backend
REPO=harmonyrestorer-backend
ALIAS=live
EXEC_ROLE=harmonyrestorer-backend-exec
KS_FUNCTION=harmonyrestorer-kill-switch
KS_ROLE=harmonyrestorer-kill-switch-exec
ALARM=harmonyrestorer-backend-busy
THROTTLE_ALARM=harmonyrestorer-backend-throttles
TOPIC=harmonyrestorer-alarms
BUDGET=harmonyrestorer-monthly
FRONTEND_ORIGIN=https://harmonyrestorer.online
ALERT_EMAIL='<the address that receives budget and alarm email>'
REGISTRY="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
TOPIC_ARN="arn:aws:sns:$REGION:$ACCOUNT_ID:$TOPIC"
WORK="$(mktemp -d)"
```

`WORK` holds the JSON files the steps write. It is outside the repository.

If a later step runs in a new shell, run this block again first. Values set by later steps (`COMMIT`, `DIGEST`, `VERSION`, `URL`) must be set again too; the steps that need them start with a check that stops when one is empty.

## 1. Budget

ADR 009 decision 13: a $20 monthly cost budget on unblended cost, so that credits do not hide spending, with email at $5, $10 and $20 of actual cost and when the forecast passes $20. A budget reports spending; it does not stop it.

```bash
cat > "$WORK/budget.json" <<EOF
{
  "BudgetName": "$BUDGET",
  "BudgetLimit": {"Amount": "20", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostTypes": {"UseBlended": false, "IncludeCredit": false, "IncludeRefund": false}
}
EOF

notification() {  # type threshold
  printf '{"Notification": {"NotificationType": "%s", "ComparisonOperator": "GREATER_THAN", "Threshold": %s, "ThresholdType": "ABSOLUTE_VALUE"}, "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "%s"}]}' "$1" "$2" "$ALERT_EMAIL"
}
printf '[%s, %s, %s, %s]\n' "$(notification ACTUAL 5)" "$(notification ACTUAL 10)" \
  "$(notification ACTUAL 20)" "$(notification FORECASTED 20)" > "$WORK/notifications.json"

"$AWS" budgets create-budget --account-id "$ACCOUNT_ID" \
  --budget "file://$WORK/budget.json" \
  --notifications-with-subscribers "file://$WORK/notifications.json"
```

The Budgets API marks `CostTypes` as deprecated and recommends `Metrics` and `FilterExpression`, but still accepts it. It is used here because its `IncludeCredit` field ("The default value is `true`") is the documented way to leave credits out; the pages read do not say how `Metrics` treats credits.

Check: `"$AWS" budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name "$BUDGET" --query "Budget.[BudgetName,BudgetLimit.Amount,CostTypes.IncludeCredit]"`

Undo: `"$AWS" budgets delete-budget --account-id "$ACCOUNT_ID" --budget-name "$BUDGET"`

## 2. Alarm email

ADR 009 decision 13: alarms send email through SNS. AWS sends a confirmation email that must be accepted before any alarm email arrives.

```bash
"$AWS" sns create-topic --name "$TOPIC" > /dev/null
"$AWS" sns subscribe --topic-arn "$TOPIC_ARN" --protocol email \
  --notification-endpoint "$ALERT_EMAIL" --query SubscriptionArn --output text
```

The second command prints `pending confirmation`. Accept the email, then check that one subscription is confirmed. An unconfirmed subscription's ARN reads `PendingConfirmation` in the list, per an example in the SNS developer guide:

`"$AWS" sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN" --query "length(Subscriptions[?SubscriptionArn!='PendingConfirmation'])"`

Undo: `"$AWS" sns delete-topic --topic-arn "$TOPIC_ARN"`

## 3. ECR repository

ADR 009 decision 11: a private repository with scan on push and immutable tags. The function's execution role carries logging only, so the repository's policy lets Lambda pull the image; Lambda's page on container images says that for a function in the same account, "Only one side needs to allow access".

```bash
"$AWS" ecr create-repository --repository-name "$REPO" \
  --image-scanning-configuration scanOnPush=true --image-tag-mutability IMMUTABLE \
  --query repository.repositoryName --output text

cat > "$WORK/ecr-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "LambdaECRImageRetrievalPolicy",
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
    "Condition": {"StringLike": {"aws:sourceArn": "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION*"}}
  }]
}
EOF
"$AWS" ecr set-repository-policy --repository-name "$REPO" \
  --policy-text "file://$WORK/ecr-policy.json" > /dev/null
```

The policy follows AWS's minimum example, with a condition added that limits it to function ARNs starting with this function's name, which covers its versions and aliases (and any other function whose name starts the same way); AWS's only `aws:sourceArn` example for Lambda image pulls uses `function:*`. On 2026-11-07, step 6 created the function after this policy, with the condition, was set. If `create-function` reports that it cannot pull the image, replace the policy with AWS's minimum example, without the condition.

The ECR API marks the repository's `imageScanningConfiguration` as being deprecated in favour of a registry-level scanning setting; the option still works, and the check below shows whether scan on push is on.

Check: `"$AWS" ecr describe-repositories --repository-names "$REPO" --query "repositories[0].[imageTagMutability,imageScanningConfiguration.scanOnPush]"`

Undo: `"$AWS" ecr delete-repository --repository-name "$REPO" --force`

## 4. Image

ADR 009 decision 11: deploy the image CI published, by the digest of its linux/amd64 image, not the multi-platform index. Pick the commit to deploy, then read that image's amd64 digest from GHCR:

```bash
COMMIT='<full commit SHA whose CI run published the image>'
docker buildx imagetools inspect "ghcr.io/jacob7choi-xyz/harmonyrestorer-backend:$COMMIT"
DIGEST='<the sha256 digest listed with Platform: linux/amd64>'
```

```bash
: "${COMMIT:?set COMMIT first}" "${DIGEST:?set DIGEST first}"
docker pull --platform linux/amd64 "ghcr.io/jacob7choi-xyz/harmonyrestorer-backend@$DIGEST"
"$AWS" ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker tag "ghcr.io/jacob7choi-xyz/harmonyrestorer-backend@$DIGEST" "$REGISTRY/$REPO:$COMMIT"
docker push "$REGISTRY/$REPO:$COMMIT" 2>&1 | grep -o 'digest: sha256:[0-9a-f]\{64\}'
docker logout "$REGISTRY" > /dev/null
```

`docker push` and `docker logout` print the registry host, which contains the account ID, so only the digest line is shown. It must equal `DIGEST`. If nothing is printed, the push failed; the check below shows whether the image arrived. On 2026-11-06 the pushed digest equalled `DIGEST`; if it ever does not, stop and compare the manifests before going on.

Check: `"$AWS" ecr describe-images --repository-name "$REPO" --query "imageDetails[].imageDigest"` lists `DIGEST`.

Tags in this repository are immutable, so pushing the same `COMMIT` tag a second time fails with `ImageTagAlreadyExistsException`.

Undo: `"$AWS" ecr batch-delete-image --repository-name "$REPO" --image-ids imageDigest="$DIGEST"`

## 5. Logs and execution role

ADR 009 decision 13 and the security design: the function's log group keeps 14 days, and the execution role may use that log group only. Lambda's logging page lists `logs:CreateLogGroup`, `logs:CreateLogStream` and `logs:PutLogEvents` for the execution role, so all three are granted, limited to this one log group. The log group is created here first so that its retention is set before the function writes to it.

```bash
"$AWS" logs create-log-group --log-group-name "/aws/lambda/$FUNCTION"
"$AWS" logs put-retention-policy --log-group-name "/aws/lambda/$FUNCTION" --retention-in-days 14

cat > "$WORK/lambda-trust.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF
cat > "$WORK/exec-logs.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
    "Resource": "arn:aws:logs:$REGION:$ACCOUNT_ID:log-group:/aws/lambda/$FUNCTION:*"
  }]
}
EOF
"$AWS" iam create-role --role-name "$EXEC_ROLE" \
  --assume-role-policy-document "file://$WORK/lambda-trust.json" --query Role.RoleName --output text
"$AWS" iam put-role-policy --role-name "$EXEC_ROLE" --policy-name write-own-logs \
  --policy-document "file://$WORK/exec-logs.json"
```

Check: `"$AWS" logs describe-log-groups --log-group-name-prefix "/aws/lambda/$FUNCTION" --query "logGroups[].[logGroupName,retentionInDays]"`

Undo: `"$AWS" iam delete-role-policy --role-name "$EXEC_ROLE" --policy-name write-own-logs`, then `"$AWS" iam delete-role --role-name "$EXEC_ROLE"`, then `"$AWS" logs delete-log-group --log-group-name "/aws/lambda/$FUNCTION"`.

## 6. Function

ADR 009 decisions 1, 3, 5 to 9. Every limit is set explicitly. `CORS_ORIGINS` is empty so that only the function URL adds CORS headers (the app then adds no CORS middleware). Reserved concurrency 1 is set right after creation; the function has no URL yet, so it has no public endpoint in between.

```bash
cat > "$WORK/function-env.json" <<'EOF'
{"Variables": {
  "CORS_ORIGINS": "",
  "CLIENT_IDENTITY": "lambda_request_context",
  "UPLOAD_DIR": "/tmp/uploads",
  "PROCESSED_DIR": "/tmp/processed",
  "MAX_UPLOAD_BYTES": "4194304",
  "MAX_ARTIFACT_BYTES": "1073741824",
  "MAX_JOBS_PER_IP": "6",
  "JOB_TTL_SECONDS": "1200",
  "MAX_TOTAL_JOBS": "24",
  "RATE_LIMIT_MAX_REQUESTS": "10",
  "RATE_LIMIT_WINDOW_SECONDS": "60",
  "DOWNLOAD_RATE_LIMIT_MAX_REQUESTS": "60",
  "DOWNLOAD_RATE_LIMIT_WINDOW_SECONDS": "3600",
  "ENABLE_DOCS": "false",
  "LOG_FORMAT": "json",
  "LOG_LEVEL": "INFO",
  "AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS": "100"
}}
EOF

: "${DIGEST:?set DIGEST first}"
"$AWS" lambda create-function --function-name "$FUNCTION" --package-type Image \
  --code ImageUri="$REGISTRY/$REPO@$DIGEST" \
  --role "arn:aws:iam::$ACCOUNT_ID:role/$EXEC_ROLE" \
  --architectures x86_64 --memory-size 3008 --timeout 120 --ephemeral-storage Size=2048 \
  --environment "file://$WORK/function-env.json" \
  --query "[FunctionName,State]" --output text
"$AWS" lambda wait function-active-v2 --function-name "$FUNCTION"
"$AWS" lambda put-function-concurrency --function-name "$FUNCTION" \
  --reserved-concurrent-executions 1 --query ReservedConcurrentExecutions
```

On 2026-11-07 Lambda accepted the empty `CORS_ORIGINS` and the `@sha256:` form of `ImageUri`. If either is refused, `","` gives the app the same empty list (`backend/app/config.py:167-171` drops blank entries), and `$REGISTRY/$REPO:$COMMIT` names the same image by a tag that is immutable in this repository (step 3).

The memory is 3,008 MB because this account refused more during the first setup (ADR 009 decision 5). If `create-function` refuses it, stop and record the error.

`AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS` repeats the image's value of 100 (`Dockerfile:94-97`). An image published from an earlier commit sets 8, or nothing before `1bf5aa5`, and the function's value overrides it. With 8, the adapter failed init on 2026-11-07 (ADR 009 decision 4).

IAM changes may not be visible everywhere at once (AWS's IAM troubleshooting page: changes including "creating or updating users, groups, roles, or policies" are subject to eventual consistency). If `create-function` reports that the role cannot be assumed right after step 5, wait a short time and run it again. The same applies to the kill-switch function in step 8.

Check: `"$AWS" lambda get-function-configuration --function-name "$FUNCTION" --query "[State,MemorySize,Timeout,EphemeralStorage.Size,Architectures[0]]"` and `"$AWS" lambda get-function-concurrency --function-name "$FUNCTION"`.

Undo: `"$AWS" lambda delete-function --function-name "$FUNCTION"`

## 7. Version and alias

ADR 009 decision 11: the function URL serves the alias `live`, and a deploy points `live` at a published version.

```bash
VERSION="$("$AWS" lambda publish-version --function-name "$FUNCTION" --query Version --output text)"
"$AWS" lambda create-alias --function-name "$FUNCTION" --name "$ALIAS" \
  --function-version "$VERSION" --query Name --output text
```

Check: `"$AWS" lambda get-alias --function-name "$FUNCTION" --name "$ALIAS" --query FunctionVersion`

Then warm the version through the alias with a signed call, so that its first start falls on this call rather than on a visitor's request; a later cold start can still fall on a visitor. On 2026-11-07 the first start of the image deployed then was ready after about 78 s (ADR 009, Precondition results). The event has the shape the function URL sends; the call needs IAM credentials, so it works before the URL exists. Repeat it after every `update-alias` in this runbook.

```bash
cat > "$WORK/health-event.json" <<'EOF'
{
  "version": "2.0",
  "rawPath": "/health",
  "rawQueryString": "",
  "headers": {"host": "warm-up"},
  "requestContext": {
    "http": {"method": "GET", "path": "/health", "protocol": "HTTP/1.1", "sourceIp": "198.51.100.7", "userAgent": "warm-up"},
    "requestId": "warm-up",
    "routeKey": "$default",
    "stage": "$default",
    "timeEpoch": 0
  },
  "isBase64Encoded": false
}
EOF
"$AWS" lambda invoke --function-name "$FUNCTION" --qualifier "$ALIAS" \
  --cli-binary-format raw-in-base64-out --cli-read-timeout 150 \
  --payload "file://$WORK/health-event.json" "$WORK/warm-up.out" \
  --query "[StatusCode,FunctionError]" --output text
tr -d '\000' < "$WORK/warm-up.out"; echo
```

Passing: `200` and `None`, and the output contains `"status":"healthy"`. The CLI used on 2026-11-07 (2.39.5) had no command for a streamed invoke, so this uses a plain invoke, which returned the response in full that day.

Undo: `"$AWS" lambda delete-alias --function-name "$FUNCTION" --name "$ALIAS"`

## 8. Kill switch

ADR 009 decision 14: an alarm on the function's summed `Duration` invokes `ops/kill_switch/handler.py`, which sets the function URL's authentication to `AWS_IAM`. The threshold is more than 900,000 ms (15 minutes) of `Duration` in one hour. CloudWatch evaluates an alarm with a period of one minute or longer "every minute", over a sliding window of the period, so this alarm can act within about a minute of the hourly sum passing the threshold, once the metric data has arrived; how long that takes was not measured. At ADR 009's worst case of up to about $1.20 per busy hour, 15 busy minutes cost about $0.30. `Duration` counts every version and alias of the function under the `FunctionName` dimension, and does not include cold start time.

The kill-switch function may update the URL configuration of this one function, named by its ARN and its alias's ARN, and write its own logs.

```bash
"$AWS" logs create-log-group --log-group-name "/aws/lambda/$KS_FUNCTION"
"$AWS" logs put-retention-policy --log-group-name "/aws/lambda/$KS_FUNCTION" --retention-in-days 14

cat > "$WORK/ks-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "lambda:UpdateFunctionUrlConfig",
      "Resource": [
        "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION",
        "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION:$ALIAS"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:$REGION:$ACCOUNT_ID:log-group:/aws/lambda/$KS_FUNCTION:*"
    }
  ]
}
EOF
"$AWS" iam create-role --role-name "$KS_ROLE" \
  --assume-role-policy-document "file://$WORK/lambda-trust.json" --query Role.RoleName --output text
"$AWS" iam put-role-policy --role-name "$KS_ROLE" --policy-name close-function-url \
  --policy-document "file://$WORK/ks-policy.json"

(cd ops/kill_switch && zip -j "$WORK/kill-switch.zip" handler.py)
"$AWS" lambda create-function --function-name "$KS_FUNCTION" --runtime python3.13 \
  --handler handler.lambda_handler --zip-file "fileb://$WORK/kill-switch.zip" \
  --role "arn:aws:iam::$ACCOUNT_ID:role/$KS_ROLE" --timeout 30 --memory-size 128 \
  --environment "Variables={TARGET_FUNCTION=$FUNCTION,TARGET_QUALIFIER=$ALIAS,ALARM_NAME=$ALARM}" \
  --query "[FunctionName,State]" --output text
"$AWS" lambda wait function-active-v2 --function-name "$KS_FUNCTION"

"$AWS" lambda add-permission --function-name "$KS_FUNCTION" --statement-id cloudwatch-alarm \
  --action lambda:InvokeFunction --principal lambda.alarms.cloudwatch.amazonaws.com \
  --source-account "$ACCOUNT_ID" \
  --source-arn "arn:aws:cloudwatch:$REGION:$ACCOUNT_ID:alarm:$ALARM" > /dev/null

"$AWS" cloudwatch put-metric-alarm --alarm-name "$ALARM" \
  --namespace AWS/Lambda --metric-name Duration \
  --dimensions Name=FunctionName,Value="$FUNCTION" \
  --statistic Sum --period 3600 --evaluation-periods 1 \
  --threshold 900000 --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$KS_FUNCTION" "$TOPIC_ARN"

"$AWS" cloudwatch put-metric-alarm --alarm-name "$THROTTLE_ALARM" \
  --namespace AWS/Lambda --metric-name Throttles \
  --dimensions Name=FunctionName,Value="$FUNCTION" \
  --statistic Sum --period 3600 --evaluation-periods 1 \
  --threshold 100 --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching --alarm-actions "$TOPIC_ARN"
```

The throttle alarm only emails. Throttles are expected while one user's job holds the environment (ADR 009, Consequences), so its threshold of more than 100 an hour is a judgment, to be adjusted from what the metric shows.

**To verify:** which resource ARN IAM checks when `lambda:UpdateFunctionUrlConfig` names an alias; the pages read do not say, so the policy lists the function's ARN and the alias's ARN. On 2026-11-07 the kill switch made the change with this policy (precondition 7), which does not show which of the two ARNs IAM matched.

Check: `"$AWS" cloudwatch describe-alarms --alarm-names "$ALARM" "$THROTTLE_ALARM" --query "MetricAlarms[].[AlarmName,StateValue,Threshold,Period]"`. The kill switch itself is tested in step 10, after the URL exists.

Undo, in this order:

```bash
"$AWS" cloudwatch delete-alarms --alarm-names "$ALARM" "$THROTTLE_ALARM"
"$AWS" lambda delete-function --function-name "$KS_FUNCTION"
"$AWS" iam delete-role-policy --role-name "$KS_ROLE" --policy-name close-function-url
"$AWS" iam delete-role --role-name "$KS_ROLE"
"$AWS" logs delete-log-group --log-group-name "/aws/lambda/$KS_FUNCTION"
```

## 9. Function URL

ADR 009 decisions 2 and 3: `NONE` authentication, `RESPONSE_STREAM`, CORS for the frontend's origin only, on the alias `live`. With the CLI, the public permission is added by hand, as two statements: `lambda:InvokeFunctionUrl` with the condition that the authentication is `NONE`, which is what the kill switch relies on, and `lambda:InvokeFunction` limited to calls through the function URL. Both follow AWS's default policy for the `NONE` auth type.

```bash
cat > "$WORK/cors.json" <<EOF
{"AllowOrigins": ["$FRONTEND_ORIGIN"], "AllowMethods": ["GET", "POST"]}
EOF
"$AWS" lambda create-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS" \
  --auth-type NONE --invoke-mode RESPONSE_STREAM --cors "file://$WORK/cors.json" \
  --query FunctionUrl --output text

"$AWS" lambda add-permission --function-name "$FUNCTION" --qualifier "$ALIAS" \
  --statement-id public-function-url --action lambda:InvokeFunctionUrl \
  --principal '*' --function-url-auth-type NONE > /dev/null
"$AWS" lambda add-permission --function-name "$FUNCTION" --qualifier "$ALIAS" \
  --statement-id public-function-url-invoke --action lambda:InvokeFunction \
  --principal '*' --invoked-via-function-url > /dev/null

URL="$("$AWS" lambda get-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS" --query FunctionUrl --output text)"
URL="${URL%/}"
```

Check: `curl -sS "$URL/health"` is expected to return 200 and `"status":"healthy"`, after a cold start on the first call.

Undo: `"$AWS" lambda delete-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS"`, then `"$AWS" lambda remove-permission --function-name "$FUNCTION" --qualifier "$ALIAS" --statement-id public-function-url` and the same for `public-function-url-invoke`.

## 10. Preconditions

ADR 009's preconditions 1 to 7. Record each result in the ADR. The frontend is not pointed at the URL until all of them pass.

**7. Kill switch, first.** Put the alarm into `ALARM` by hand; AWS says the configured actions run when the state changes:

```bash
: "${URL:?set URL first}"
"$AWS" cloudwatch set-alarm-state --alarm-name "$ALARM" --state-value ALARM --state-reason "runbook test"
for _ in $(seq 1 24); do
  auth="$("$AWS" lambda get-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS" --query AuthType --output text)"
  [ "$auth" = AWS_IAM ] && break
  sleep 5
done
echo "$auth"
curl -sS -o /dev/null -w '%{http_code}\n' "$URL/health"
```

The actions run on the change to `ALARM`. Passing: `AuthType` is `AWS_IAM` and the request is refused. On 2026-11-07 the refusal was `403` with `{"Message":"Forbidden"}`, but the first request after `AuthType` read `AWS_IAM` still got 200 and the next, 5 s later, got 403; if the code above prints 200, run the `curl` line again after a few seconds. Then reopen the URL and confirm service is back:

```bash
"$AWS" lambda update-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS" \
  --auth-type NONE --query AuthType --output text
"$AWS" cloudwatch set-alarm-state --alarm-name "$ALARM" --state-value OK --state-reason "runbook test done"
curl -sS -o /dev/null -w '%{http_code}\n' "$URL/health"
"$AWS" lambda get-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS" \
  --query "[AuthType,InvokeMode,Cors.AllowOrigins[0]]" --output text
```

On 2026-11-07 the first request after reopening got 403 and the next, 5 s later, got 200; if the code above prints 403, run the `curl` line again after a few seconds. The last command checks that changing only the authentication, as the kill switch and the reopen both do, kept the invoke mode and the CORS origin. A read of the same fields after the test on 2026-11-07 showed `NONE`, `RESPONSE_STREAM`, and the frontend's origin with the methods GET and POST.

The alarm's actions run only when its state changes to `ALARM`. After a real trip, reopen the URL only once the hourly sum is back under the threshold and the alarm reads `OK` (`"$AWS" cloudwatch describe-alarms --alarm-names "$ALARM" --query "MetricAlarms[0].StateValue"`); while it stays in `ALARM`, it cannot close the URL again.

**1. Client identity.** Set `LOG_LEVEL=DEBUG` for the check. `update-function-configuration --environment` replaces every variable, so it is given the whole file:

```bash
sed -i '' 's/"LOG_LEVEL": "INFO"/"LOG_LEVEL": "DEBUG"/' "$WORK/function-env.json"
"$AWS" lambda update-function-configuration --function-name "$FUNCTION" \
  --environment "file://$WORK/function-env.json" --query LastUpdateStatus --output text
"$AWS" lambda wait function-updated-v2 --function-name "$FUNCTION"
VERSION="$("$AWS" lambda publish-version --function-name "$FUNCTION" --query Version --output text)"
"$AWS" lambda update-alias --function-name "$FUNCTION" --name "$ALIAS" \
  --function-version "$VERSION" --query FunctionVersion --output text
```

(`sed -i ''` is the macOS form; on Linux it is `sed -i`.) Warm the new version as in step 7. Then upload once with forged headers, and make one request from a second network. A phone on its cellular network, with Wi-Fi off, can open `$URL/api/v1/download/00000000-0000-4000-8000-000000000000` in a browser; the job does not exist, so it gets 404, but the request still passes the rate-limit check that logs the address.

```bash
curl -sS -F "file=@frontend/public/sample-noisy.wav;type=audio/wav" \
  -H 'X-Forwarded-For: 203.0.113.9' -H 'x-amzn-request-context: {"http":{"sourceIp":"203.0.113.9"}}' \
  "$URL/api/v1/denoise" | jq .status
"$AWS" logs filter-log-events --log-group-name "/aws/lambda/$FUNCTION" \
  --filter-pattern '"Rate-limit check"' --query "events[].message"
```

Passing: the two networks log different addresses, and the forged request logs the real address, not `203.0.113.9`. Afterwards, set `LOG_LEVEL` back to `INFO` with the same commands, swapping `DEBUG` and `INFO` in the `sed` line, and warm the new version.

**2. Upload size.** Make a WAV of exactly 4 MiB and one byte larger, and upload both:

```bash
uv run --frozen python -I - <<'EOF'
import struct
for name, size in (("exact.wav", 4194304), ("over.wav", 4194306)):
    data = size - 44
    header = (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVEfmt " +
              struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) +
              b"data" + struct.pack("<I", data))
    open(name, "wb").write(header + b"\0" * data)
EOF
for f in exact.wav over.wav; do
  curl -sS -o /dev/null -w "$f %{http_code}\n" -F "file=@$f;type=audio/wav" "$URL/api/v1/denoise"
done
```

Passing: `exact.wav` gets 200 and `over.wav` 413. (`over.wav` is two bytes over, to keep the 16-bit samples whole.)

**3. Streaming.** Upload a 595 s file under 4 MiB, then download the WAV. It is a few seconds under the 600 s cap because uploads longer than the cap are refused (`backend/app/services/audio_decode.py:246`) and an MP3 encoder can add padding:

```bash
ffmpeg -y -f lavfi -i anoisesrc=d=595:c=pink:r=16000 -ac 1 -b:a 48k long.mp3
JOB="$(curl -sS -F "file=@long.mp3;type=audio/mpeg" "$URL/api/v1/denoise" | jq -r .job_id)"
curl -sS -o long-restored.wav -w '%{http_code} %{size_download}\n' "$URL/api/v1/download/$JOB"
```

Passing: 200 and about 19.0 MB (595 s at 16 kHz, 16-bit mono, plus a 44-byte header), more than the 6 MB a buffered response allows.

**4. Readable 429.** While a long upload runs, ask for a status with the frontend's origin:

```bash
curl -sS -F "file=@long.mp3;type=audio/mpeg" "$URL/api/v1/denoise" > /dev/null &
sleep 3
curl -sS -D - -o /dev/null -H "Origin: $FRONTEND_ORIGIN" \
  "$URL/api/v1/status/00000000-0000-4000-8000-000000000000"
wait
```

Passing: the status request gets 429 with an `access-control-allow-origin` header naming the frontend's origin. AWS says the URL adds its CORS headers "to all responses through the function URL" and does not mention throttled responses by name, so this check is the evidence.

**5. Init.** `"$AWS" logs filter-log-events --log-group-name "/aws/lambda/$FUNCTION" --filter-pattern '"Init Duration"' --query "events[].message"`. In the logs read on 2026-11-07, `Init Duration` appeared on the REPORT line of a cold start whose init finished within the init phase, and an `INIT_REPORT` line with `Status: timeout` marked one that did not; init then finished during the first request. Passing, as amended on 2026-11-07: once the image is cached, a cold start answers its first request within 20 s. To see it, publish a new version and point `live` at it (the commands in precondition 1, with any change), so that the next call needs a new environment, and time the warm-up call from step 7. The first start of a new image is not covered by this; the warm-up takes it. The 20 s was chosen after four such starts, on one image, were observed (ADR 009, Precondition results).

**6. Memory.** Passing: step 6's `create-function` accepted the memory it sets, and `get-function-configuration` reports it.

## 11. Deploys from GitHub

ADR 009 decisions 11 and 12: after CI passes on `main`, the `deploy` job in `.github/workflows/ci.yml` is set up to run `scripts/deploy_lambda.sh` once a reviewer approves it. It signs in with GitHub's OIDC token, so no AWS key is stored in this repository's secrets for it. This step sets up the AWS side and the GitHub environment it runs in. The environment is `aws-backend`, not `Production`: the repository's `Production` environment has the frontend deployments that `vercel[bot]` records, and the rules below are kept off it.

This part creates the OIDC provider and a rule that keeps the 5 most recent images in the ECR repository. AWS's CLI reference says the thumbprint is optional: "If it is not included, IAM will retrieve and use the top intermediate certificate authority (CA) thumbprint".

```bash
"$AWS" iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com --query OpenIDConnectProviderArn --output text > /dev/null

cat > "$WORK/ecr-lifecycle.json" <<'EOF'
{"rules": [{
  "rulePriority": 1,
  "description": "Keep the 5 most recent images",
  "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 5},
  "action": {"type": "expire"}
}]}
EOF
"$AWS" ecr put-lifecycle-policy --repository-name "$REPO" \
  --lifecycle-policy-text "file://$WORK/ecr-lifecycle.json" --query repositoryName --output text
```

> **Note:** The rule deletes images older than the 5 most recent. Whether Lambda can still start a version whose image was deleted was not checked, so moving `live` back to such a version, by the script or by hand, may fail.

The deploy role trusts only a token for this repository's `aws-backend` environment. Its permissions name the one ECR repository and the one function, apart from `ecr:GetAuthorizationToken`, which AWS's example grants on `*`. The ECR actions are the ones AWS lists for pushing an image, with `GetDownloadUrlForLayer` and `GetRepositoryPolicy`, which Lambda's container image page lists for the identity that creates a function, and `DescribeImages` for the script's check. `InvokeFunction` covers every version and alias of the function, so the script can warm a new version before the alias moves to it. It cannot change the function's configuration, its reserved concurrency, its URL or the kill switch.

```bash
DEPLOY_ROLE=harmonyrestorer-backend-deploy
cat > "$WORK/deploy-trust.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::$ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {"StringEquals": {
      "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
      "token.actions.githubusercontent.com:sub": "repo:jacob7choi-xyz/harmonyrestorer-v1:environment:aws-backend"
    }}
  }]
}
EOF
cat > "$WORK/deploy-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
    {
      "Sid": "EcrPushAndRead",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer", "ecr:DescribeImages", "ecr:GetRepositoryPolicy"
      ],
      "Resource": "arn:aws:ecr:$REGION:$ACCOUNT_ID:repository/$REPO"
    },
    {
      "Sid": "LambdaDeploy",
      "Effect": "Allow",
      "Action": [
        "lambda:UpdateFunctionCode", "lambda:GetFunction", "lambda:GetFunctionConfiguration",
        "lambda:PublishVersion", "lambda:GetAlias", "lambda:UpdateAlias"
      ],
      "Resource": [
        "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION",
        "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION:$ALIAS"
      ]
    },
    {
      "Sid": "WarmUpVersions",
      "Effect": "Allow",
      "Action": "lambda:InvokeFunction",
      "Resource": "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION:*"
    }
  ]
}
EOF
"$AWS" iam create-role --role-name "$DEPLOY_ROLE" \
  --assume-role-policy-document "file://$WORK/deploy-trust.json" --query Role.RoleName --output text
"$AWS" iam put-role-policy --role-name "$DEPLOY_ROLE" --policy-name deploy-backend \
  --policy-document "file://$WORK/deploy-policy.json"
```

This part sets up the GitHub environment, with the repository owner as its required reviewer, deployments from `main` only, and the role's ARN as a secret. It is a secret rather than a variable because the log lists a step's inputs at the start of the step: the first deploy read it from a variable, and that line showed the account ID unmasked. GitHub's documentation says secrets are masked in logs. The ARN goes to `gh` without being printed. `gh api user` names the signed-in GitHub account, which is expected to be `jacob7choi-xyz`.

```bash
GH_REPO=jacob7choi-xyz/harmonyrestorer-v1
REVIEWER_ID="$(gh api user --jq .id)"
gh api -X PUT "repos/$GH_REPO/environments/aws-backend" --input - > /dev/null <<EOF
{"reviewers": [{"type": "User", "id": $REVIEWER_ID}], "prevent_self_review": false,
 "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
EOF
gh api -X POST "repos/$GH_REPO/environments/aws-backend/deployment-branch-policies" \
  -f name=main -f type=branch > /dev/null
gh secret set AWS_DEPLOY_ROLE_ARN --env aws-backend --repo "$GH_REPO" \
  --body "arn:aws:iam::$ACCOUNT_ID:role/$DEPLOY_ROLE"
```

`prevent_self_review` is false because the reviewer is also the person who pushes to `main`.

Check: `"$AWS" iam get-role --role-name "$DEPLOY_ROLE" --query "Role.AssumeRolePolicyDocument.Statement[0].Condition"`, and `gh api "repos/$GH_REPO/environments/aws-backend" --jq '[.protection_rules[].type]'`.

**Approving a deploy.** After a push to `main`, the run's `deploy` job waits. In the run's page on GitHub, "Review deployments" approves or rejects it. A push that changes only documentation still runs the image and publish jobs and offers a deploy, which can be rejected. When the job's smoke test fails, the job moves `live` back to the version it was on and fails; the job's log says which version that is.

**When the digest check fails.** The push has then left the commit's tag on another digest, and tags in this repository cannot be overwritten, so a re-run fails until that tag is deleted. The script stops before it updates the function's code, so that deploy did not point any version at the image under the tag. Check which digest the tag names with `"$AWS" ecr describe-images --repository-name "$REPO" --image-ids imageTag=<commit> --query "imageDetails[0].imageDigest"`, then delete it with `"$AWS" ecr batch-delete-image --repository-name "$REPO" --image-ids imageTag=<commit>`.

**Moving `live` by hand.** `"$AWS" lambda update-alias --function-name "$FUNCTION" --name "$ALIAS" --function-version <version>`, then warm it as in step 7.

Undo, in this order. Deleting the environment also deletes its secret.

```bash
gh api -X DELETE "repos/$GH_REPO/environments/aws-backend"
"$AWS" iam delete-role-policy --role-name "$DEPLOY_ROLE" --policy-name deploy-backend
"$AWS" iam delete-role --role-name "$DEPLOY_ROLE"
"$AWS" ecr delete-lifecycle-policy --repository-name "$REPO" > /dev/null
"$AWS" iam delete-open-id-connect-provider \
  --open-id-connect-provider-arn "arn:aws:iam::$ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
```

## 12. Taking it down

In reverse order: the GitHub environment, the deploy role, the ECR rule and the OIDC provider (step 11), the function URL and its two permission statements (step 9), the alarms, the kill-switch function, its role and log group (step 8), the alias (step 7), the function (step 6), the execution role and log group (step 5), the images and repository (steps 3 and 4), the SNS topic (step 2) and the budget (step 1). Each step's undo command is given above. Removing the function URL alone (step 9's undo) removes the public endpoint and keeps everything else.
