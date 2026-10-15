# MentorMind campaign rules

## What MentorMind is, and what we may say

- A web app where people talk with AI mentors built from what real historical figures wrote. Live site: mentormindhq.com. Handle on every platform: @mentormindhq. Not open to the public yet. Every reel ends "Follow for launch day".
- Always say "AI mentors, built from what they wrote". Never imply the voices are the real people.
- The narrator is an AI voice and the portraits and many images are AI illustrations. Label posts as AI-generated on Instagram, TikTok and YouTube.
- Never invent screens or mentor replies. Use only real app screens, and never show a reply coming back.
- Never say the AI "understands" you. What is true: the writers carried the same things and wrote it down.
- Emotional days carry a safety line: "Not therapy. Not professional advice. In a crisis, contact local emergency services." (Day 4 had it planned on the closing card.)
- Quotes only from the verified list in the Project file `quotes-by-theme.md`, whole sentences, never extended past the safe text. Marcus Aurelius uses the public translation in the app ("thy nature requireth").
- Known false claims on the landing page, to fix before launch: it says mentors use only what real figures wrote (false for Tubman, whose source is a biography) and that every conversation is remembered (the code keeps the last 3 sessions).
- Historical note: Day 9's printing press is a stylised tribute. A real press printed a whole page at once with a screw-driven platen. Say "a printer's tribute" in the caption.

## How we work with Enoch

- Deliver whole files, plus a precise Claude Code deploy prompt. Never partial edits.
- Ask before building anything that is a real design or behaviour choice. One change at a time. He confirms it works before the next.
- Be honest over flattering. Keep it concise. No em dashes.
- He cannot see option tables or button cards. Write choices as a short plain list in the message, and let him reply with a letter or a name.
- He loves animation and motion design. He wants each day to feel clearly different, not templated.
- The founder's own words on camera are his. We design, cut and sync. We do not write what he says on camera.

## Design rules we learned

- Variety check before every day: setting, time of day, light and colour, camera feel, scale, indoor or outdoor. At least four must differ from the day before.
- Do not rush. Give each moment room to be read. Use gaps, not speed, for pacing. Do not slow a voice line below about 0.88 speed.
- Too much silence in a voice is also bad. Tight gaps (0.2 to 0.3 s) inside a line, longer beats only where the picture is moving.
- Text must never cover faces or hands. Keep it in the middle 80% of the width. Put a soft scrim behind text over bright or busy areas. Keep text clear of the corner brand mark.
- Brand mark (logo and @mentormindhq) bottom left on every frame. Closing card: logo, "Opening soon.", "Follow for launch day." Days 4 to 8 add the "AI mentors, built from what they wrote" line.
- Format: 1080 x 1920, 24 fps, about -16 LUFS. Sound stays minimal: voice, a very quiet wind bed, and only sounds that belong to the picture. No whooshes.
- Inspect AI clips frame by frame before using them. Day 3 line 3 had a duplicated man at the end.
- AI source images often have a white edge. Trim it before use.
- Mentor portraits must plausibly look like the person. The first Douglass portrait did not, and was redone.
- The speech recogniser (asr_check.py) is biased toward American speech and cannot judge "AI mentors". Ask Enoch to listen.
- Render limit: about 300 seconds per tool call, one core.

## Voices

Kokoro TTS, full-size model. Isabella (bf_isabella), Michael (am_michael), Nicole (af_nicole). Delivery is shaped per line with small pitch and loudness changes. A real human voice is the biggest remaining quality lever.
