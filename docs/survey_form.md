# Blind listening survey - how to build the Google Form

1. Run `python -m scripts.make_survey_clips`. It writes `outputs/survey_clips/clip_A.wav` to `clip_F.wav`
   (15 s each, equal loudness, neutral names, already shuffled). The answer key is in
   `docs/survey_clips_manifest.md` - never show it to participants.
2. Upload the six WAVs to Google Drive (share: anyone with the link) and paste each link into its own form section.
3. For every clip create exactly these two **required** questions (the analysis script matches on the titles):
   - `Clip A: Human or machine?` - multiple choice: `Human` / `Machine`
   - `Clip A: How musical is this clip? (1-5)` - linear scale 1 to 5
   (repeat for Clip B ... Clip F)
4. Collect at least 20 responses, export the responses as CSV, then run
   `python -m scripts.analyze_survey responses.csv`.
   It writes `docs/survey_results.md` and `outputs/survey_results.json`.
