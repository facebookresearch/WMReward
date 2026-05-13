# Project Page — `docs/`

This folder is the GitHub Pages source for the **WMReward** project page
(*Inference-time Physics Alignment of Video Generative Models with Latent World Models*, CVPR 2026).

## Enable GitHub Pages

In the repo settings → **Pages**, set:

- **Source:** Deploy from a branch
- **Branch:** `main` / `/docs`

The site will be served at `https://facebookresearch.github.io/WMReward/`.

## Structure

```
docs/
├── index.html
├── README.md
└── static/
    ├── css/style.css
    ├── js/script.js
    ├── images/
    │   ├── favicon.svg
    │   ├── method.png          ← TODO: add method diagram
    │   └── quantitative.png    ← TODO: add results figure
    └── videos/
        ├── teaser.mp4          ← TODO: add hero teaser
        ├── result1.mp4         ← TODO: carousel item 1
        ├── result2.mp4         ← TODO: carousel item 2
        ├── result3.mp4         ← TODO: carousel item 3
        └── result4.mp4         ← TODO: carousel item 4
```

## Placeholders to fill before publishing

- Author names, affiliations, and personal links in `index.html`
- Paper / arXiv / supplementary / poster URLs (currently `href="#"`)
- YouTube `VIDEO_ID` in the `<iframe>` once the CVPR video is uploaded
- BibTeX (final author list, page numbers if/when available)
- Replace `method.png`, `quantitative.png`, and all `.mp4` files in `static/`

## Credits

Template adapted from
[Eliahu Horwitz's Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
