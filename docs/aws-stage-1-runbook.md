# Runbook: Set Up Stage 1 on AWS

This runbook carries out ADR 009 (`docs/adr/009-pay-per-use-hosting.md`) in an AWS account, one step at a time, with the AWS CLI. Each step says why it exists, gives the commands, a read-only check, and how to undo it. Nothing here is automated: an operator runs each step and reads its output before the next.

> **Note:** Several commands print ARNs, which contain the account ID. The commands below use `--query` to print only the fields needed, or send output to `/dev/null`. Keep the account ID out of anything committed or shared.

> **Note:** The function URL is created last (step 9), after the kill switch exists (step 8), and the kill switch is the first thing tested once the URL exists (step 10). Until step 9 the function has no public endpoint.

## 0. Before starting

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
WORK="$(mktemp -d)"
```

`WORK` holds the JSON files the steps write. It is outside the repository.

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
TOPIC_ARN="$("$AWS" sns create-topic --name "$TOPIC" --query TopicArn --output text)"
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

The policy follows AWS's minimum example, with a condition added that limits it to function ARNs starting with this function's name, which covers its versions and aliases (and any other function whose name starts the same way); AWS's only `aws:sourceArn` example for Lambda image pulls uses `function:*`. **To verify:** that the condition matches Lambda's pulls in the same account. If step 6's `create-function` reports that it cannot pull the image, replace the policy with AWS's minimum example, without the condition.

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
docker pull --platform linux/amd64 "ghcr.io/jacob7choi-xyz/harmonyrestorer-backend@$DIGEST"
"$AWS" ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker tag "ghcr.io/jacob7choi-xyz/harmonyrestorer-backend@$DIGEST" "$REGISTRY/$REPO:$COMMIT"
docker push "$REGISTRY/$REPO:$COMMIT" | grep -o 'digest: sha256:[0-9a-f]\{64\}'
docker logout "$REGISTRY" > /dev/null
```

`docker push` and `docker logout` print the registry host, which contains the account ID, so only the digest line is shown. It must equal `DIGEST`. **To verify:** that pushing a single-platform image pulled by digest keeps the same digest; if it does not, stop and compare the manifests before going on.

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
  "LOG_LEVEL": "INFO"
}}
EOF

"$AWS" lambda create-function --function-name "$FUNCTION" --package-type Image \
  --code ImageUri="$REGISTRY/$REPO@$DIGEST" \
  --role "arn:aws:iam::$ACCOUNT_ID:role/$EXEC_ROLE" \
  --architectures x86_64 --memory-size 3538 --timeout 120 --ephemeral-storage Size=2048 \
  --environment "file://$WORK/function-env.json" \
  --query "[FunctionName,State]" --output text
"$AWS" lambda wait function-active-v2 --function-name "$FUNCTION"
"$AWS" lambda put-function-concurrency --function-name "$FUNCTION" \
  --reserved-concurrent-executions 1 --query ReservedConcurrentExecutions
```

**To verify:** that Lambda accepts an empty value for `CORS_ORIGINS`; the API reference states no minimum length. If it does not, `","` gives the app the same empty list (`backend/app/config.py:154-158` drops blank entries). Also to verify: that `ImageUri` accepts the `@sha256:` digest form; the reference says only "URI of a container image in the Amazon ECR registry". If it is refused, use `$REGISTRY/$REPO:$COMMIT`, whose tag is immutable in this repository (step 3).

If `create-function` refuses 3,538 MB, the account's memory quota may be lower (ADR 009 precondition 6); stop and record the error.

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

**To verify:** which resource ARN IAM checks when `lambda:UpdateFunctionUrlConfig` names an alias; the pages read do not say, so the policy lists the function's ARN and the alias's ARN. Precondition 7 (step 10) shows whether the kill switch can make the change.

Check: `"$AWS" cloudwatch describe-alarms --alarm-names "$ALARM" "$THROTTLE_ALARM" --query "MetricAlarms[].[AlarmName,StateValue,Threshold,Period]"`. The kill switch itself is tested in step 10, after the URL exists.

Undo: `"$AWS" cloudwatch delete-alarms --alarm-names "$ALARM" "$THROTTLE_ALARM"`, `"$AWS" lambda delete-function --function-name "$KS_FUNCTION"`, then the role policy, the role and the log group as in step 5.

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

URL='<the URL printed above, without a trailing slash>'
```

Check: `curl -sS "$URL/health"` is expected to return 200 and `"status":"healthy"`, after a cold start on the first call.

Undo: `"$AWS" lambda delete-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS"`, then `"$AWS" lambda remove-permission --function-name "$FUNCTION" --qualifier "$ALIAS" --statement-id public-function-url` and the same for `public-function-url-invoke`.

## 10. Preconditions

ADR 009's preconditions 1 to 7. Record each result in the ADR. The frontend is not pointed at the URL until all of them pass.

**7. Kill switch, first.** Put the alarm into `ALARM` by hand; AWS says the configured actions run when the state changes:

```bash
"$AWS" cloudwatch set-alarm-state --alarm-name "$ALARM" --state-value ALARM --state-reason "runbook test"
sleep 30
"$AWS" lambda get-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS" --query AuthType
curl -sS -o /dev/null -w '%{http_code}\n' "$URL/health"
```

The forced state returns to the alarm's real state within seconds; the actions run on the change to `ALARM`. Passing: `AuthType` is `AWS_IAM` and the request is refused. The pages read do not give the status code for an unsigned request to an `AWS_IAM` URL, so record the code that comes back. Then reopen the URL and confirm service is back:

```bash
"$AWS" lambda update-function-url-config --function-name "$FUNCTION" --qualifier "$ALIAS" \
  --auth-type NONE --query AuthType --output text
"$AWS" cloudwatch set-alarm-state --alarm-name "$ALARM" --state-value OK --state-reason "runbook test done"
curl -sS -o /dev/null -w '%{http_code}\n' "$URL/health"
```

**1. Client identity.** Set `LOG_LEVEL=DEBUG` for the check by editing `function-env.json` and running `update-function-configuration --environment` with the whole file, since that call replaces every variable, then publish a version and point `live` at it. Upload once from two networks (for example a home connection and a phone hotspot), and once with forged headers:

```bash
curl -sS -F "file=@frontend/public/sample-noisy.wav;type=audio/wav" \
  -H 'X-Forwarded-For: 203.0.113.9' -H 'x-amzn-request-context: {"http":{"sourceIp":"203.0.113.9"}}' \
  "$URL/api/v1/denoise" | jq .status
"$AWS" logs filter-log-events --log-group-name "/aws/lambda/$FUNCTION" \
  --filter-pattern '"Rate-limit check"' --query "events[].message"
```

Passing: the two networks log different addresses, and the forged request logs the real address, not `203.0.113.9`. Set `LOG_LEVEL` back to `INFO` the same way afterwards.

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
ffmpeg -f lavfi -i anoisesrc=d=595:c=pink:r=16000 -ac 1 -b:a 48k long.mp3
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

**5. Init.** `"$AWS" logs filter-log-events --log-group-name "/aws/lambda/$FUNCTION" --filter-pattern '"Init Duration"' --query "events[].message"`. `Init Duration` appears on the REPORT line of cold starts only. Passing: every init duration found is under 10,000 ms.

**6. Memory.** Passing: step 6's `create-function` accepted 3,538 MB, and `get-function-configuration` reports it.

## 11. Taking it down

In reverse order: the function URL and its two permission statements (step 9), the alarms, the kill-switch function, its role and log group (step 8), the alias (step 7), the function (step 6), the execution role and log group (step 5), the images and repository (steps 3 and 4), the SNS topic (step 2) and the budget (step 1). Each step's undo command is given above. Removing the function URL alone (step 9's undo) removes the public endpoint and keeps everything else.
