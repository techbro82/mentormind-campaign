# MentorMind campaign pack

Everything needed to re-render any day of the social campaign, even after the Claude sandbox resets.

## Restore (about two minutes)

1. Start a new Claude chat in the MentorMind Project and upload this folder (or the zip).
2. Ask Claude to run `sh setup_restore.sh`. It copies the scripts, fonts, voices and configs back to the paths the scripts expect, and downloads the voice model.
3. For Days 1 to 3 only, also restore the source clips from `MentorMind-Day1-3-source-clips.zip` into `/home/claude/clips`, `clips2` and `clips3`. These are Enoch's own AI clips. They also live in his GitHub repo `mm-temp-clips`, so that repo must not be deleted until the campaign is finished.

## Paths the scripts expect

- Scripts: `/home/claude/social/`
- Voices and timings: `/home/claude/voice/` (`dayNN_voice.json` and `dayNN_voice_raw.wav`)
- Fonts: `/tmp/fonts/` (Inter, Cormorant Garamond, Cormorant Garamond Italic)
- Project files (always present in the Project): `/mnt/project/` (logos, the 12 portraits, the new Douglass portrait, the app screens, the three Day 8 stills)
- Outputs: `/mnt/user-data/outputs/social/dayNN/`

## How to re-render each day

Each render takes roughly 130 to 190 seconds on one core, so run one per tool call (the limit is about 300 seconds).

- Days 1 to 3 (cinematic clips): `python3 assemble_day.py <config.json>`. Configs are in `configs/`. Day 1 uses `day01_real.json`. Day 3 line 3 must start at 2.0 s of its clip, because after about 5.5 s the clip shows a second copy of the man.
- Day 4 (chat screen): `python3 render_chat_day4.py out.mp4 [still times]`
- Day 5 (floating phone, real screens): `python3 render_phone_day5.py out.mp4`
- Day 6 (framed wall): `python3 render_roll_day6.py out.mp4` (uses `portrait-*.jpeg` and the new Douglass file)
- Day 7 (open book): `python3 render_book_day7.py out.mp4` (needs `book_lib.py`)
- Day 8 (stone): `python3 render_stone_day8.py out.mp4` (needs the three stills named in the script)
- Day 9 (printing press): `python3 render_press_day9.py out.mp4`

The optional second argument is a comma list of times in seconds, for which the script saves check stills as `_still_N_<time>.png`.

## Making a new voice

`python3 voice/make_voice.py spec.json` (see the top of that file for the format).

## Rules and history

- `campaign-rules.md`: what we decided and what we never claim.
- `campaign-log.md`: every day, its files, its settings and what is still open.
