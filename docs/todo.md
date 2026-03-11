# To Do

## Research Reading

- [ ] Re-read Kiranyaz 2022 cover-to-cover -- check if our training hyperparameters and loss functions match their recommendations
- [ ] Read Fonseca 2020 noise generation section -- check if we're missing degradation types (wow/flutter, speed drift)
- [ ] Skim Moliner & Valimaki 2022 -- closest competitor, benchmark to beat, two-stage refinement idea
- [ ] Skim HiFi-GAN loss design -- multi-scale discriminator + feature matching loss if our audio quality needs sharpening
- [ ] Skim Noise2Noise -- technique for training on real historical recordings without clean references

## After Benchmark (see docs/benchmarks.md for results)

- [ ] Benchmark UVR on same dataset for head-to-head comparison
- [ ] Listen to restored samples side-by-side (OpGAN vs UVR vs noisy original)
- [ ] If OpGAN wins: swap it into the production backend
- [ ] If OpGAN loses: review Kiranyaz hyperparameters, consider longer training or loss function changes
- [ ] Test on out-of-distribution audio (live recordings, different genres, real-world noise)

## Future Enhancements

- [ ] Try two-stage approach (coarse OpGAN denoise -> fine refinement) inspired by Moliner
- [ ] Borrow HiFi-GAN discriminator architecture if adversarial loss isn't producing sharp audio
- [ ] Explore Noise2Noise training on real historical recordings (no clean references needed)
- [ ] Frontend tests (vitest/jest setup)
- [ ] Duration validation for MP3/M4A/AAC on backend (currently frontend-only)
