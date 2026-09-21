# ACECQA HTML presentation

Open `index.html` in a modern browser. The deck and all four Plotly explorers are self-contained and work offline.

## Presentation controls

- Arrow keys, Page Up/Page Down, or Space: move between slides.
- `Home`: first slide.
- `End`: final main slide.
- `N`: toggle speaker notes.
- `F`: toggle full screen.
- `Esc`: close notes, an enlarged figure, or an interactive explorer.

The main presentation contains 17 slides designed for an approximately eight-minute video. Four appendix slides follow the conclusion and are available for questions.

Interactive explorer buttons open each standalone HTML analysis in a full-window layer. The same pages can be opened in a separate tab from that layer. Static figures can be clicked to enlarge them.

## Submission structure

```text
html_ppt/
|-- index.html
|-- styles.css
|-- deck.js
|-- README.md
`-- assets/
    |-- images/
    `-- interactive/
```

All paths are relative to `html_ppt`, so the folder can be submitted or moved as one unit.
