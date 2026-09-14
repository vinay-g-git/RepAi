# RepAI — The Future of Personal Training
> "AI that trains with you."

Features:
- Page 1: Strict landing page (Title, quote, Explore Now button)
- Page 2: Dedicated pages/workspaces for each feature + All Features Hub
- AI Gym Trainer (Webcam + Video Upload backup with live voice cues)
- AI Dietician & Meal Photo Scanner (Dynamic next-meal protein rebalancer)
- Behavioral Habit Tracker, Virtual Gym Buddy, Performance Analyzer, Gym Finder
- Sun/Moon Light and Dark theme toggle with hardened dropdown contrast

## Gym Finder setup
The Gym Finder shows real gyms via the Google Places API when configured,
and otherwise falls back to a clearly labeled DEMO dataset — it never shows
fabricated businesses as if they were real. To enable real results:
```
export GOOGLE_PLACES_API_KEY="AIza..."
```
Note: the Places API does not expose real monthly membership prices, so
real listings show a rough Google price tier instead of a dollar figure;
only DEMO listings show a (fictional) monthly price.

## Virtual Gym Buddy setup
The Gym Buddy is a real AI chat (via the same Anthropic API used for the
meal photo scanner) — it requires `ANTHROPIC_API_KEY` to be set on the
backend; the key is never sent to or exposed in the browser. If it isn't
set, the chat returns a clear "AI coach isn't configured" error instead of
a scripted reply. Conversation history is kept client-side for the current
browser session only (not persisted, not sent anywhere except back to this
backend as context for the next message).