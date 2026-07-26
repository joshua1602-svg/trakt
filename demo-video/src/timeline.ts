/**
 * The storyboard, as data.
 *
 * One place controls every scene's length and order, every burned-in caption and
 * every narration line. The composition, the captions track and the generated
 * voice-over script all read this table, so they cannot drift apart.
 *
 * Frames, not seconds, because the storyboard is frame-precise: 2,700 frames at
 * 30fps, cut 420 / 720 / 600 / 600 / 360. A unit test asserts the total and the
 * per-scene boundaries.
 *
 * On-screen COPY lives here. On-screen FIGURES do not — those are read from the
 * fixtures at render time (see src/data/fixtures.ts). Nothing in this file states
 * a number the pipeline produced.
 */

export const FPS = 30;
export const TOTAL_FRAMES = 2700;

export interface Caption {
  /** Frames from the start of the SCENE. */
  at: number;
  /** Frames the caption stays up. */
  hold: number;
  /** Max two lines, and it must carry the argument with the sound off. */
  text: string;
}

export interface SceneSpec {
  id: string;
  /** Scene number as the storyboard names it. */
  number: number;
  title: string;
  frames: number;
  captions: Caption[];
  /** The voice-over paragraph for this scene (UK English). */
  narration: string;
}

export const SCENES: SceneSpec[] = [
  {
    id: "cost",
    number: 1,
    title: "The cost",
    frames: 420,
    // No caption over a display claim, and none over the opening line — that line is
    // already burned-in body copy doing the caption's job. A caption underneath either
    // one just competes with it.
    captions: [
      { at: 96, hold: 132, text: "Every month-end, someone rebuilds the mapping by hand." },
    ],
    narration:
      "You bought a back book. It didn't come with your data model. It arrived as a " +
      "servicer extract that looks nothing like your origination system, and every " +
      "month-end someone rebuilds the mapping by hand. Five days of month-end, every " +
      "month, for every portfolio.",
  },
  {
    id: "onboard",
    number: 2,
    title: "Onboarded in under 48 hours",
    frames: 720,
    captions: [
      { at: 6, hold: 132, text: "Trakt onboards a portfolio once." },
      {
        at: 144,
        hold: 144,
        text: "Agents profile the source and map it to one canonical model.",
      },
      {
        at: 300,
        hold: 144,
        text: "Thirty-nine fields mapped. Thirty-four specific to this book.",
      },
      {
        at: 456,
        hold: 144,
        text: "Two referred to a human — it knows what it does not know.",
      },
    ],
    narration:
      "Trakt onboards a portfolio once, in under forty-eight hours. Agents profile the " +
      "source, map it to a canonical model and apply your lending rules. Thirty-nine " +
      "fields mapped, thirty-four decisions specific to your book — and two referred to " +
      "a human, because the platform knows what it doesn't know. Approved once, applied " +
      "unchanged every period after.",
  },
  {
    id: "dataset",
    number: 3,
    title: "One dataset, every output",
    frames: 600,
    captions: [
      { at: 12, hold: 108, text: "One governed dataset across both source portfolios." },
      {
        at: 132,
        hold: 168,
        text: "Your regulatory submission, your investor pack, your management information.",
      },
      { at: 312, hold: 108, text: "Every row provenance-stamped." },
    ],
    narration:
      "From there, one governed dataset. One point nine six billion across eleven " +
      "thousand and thirty-five loans, two source portfolios, every row " +
      "provenance-stamped. Your regulatory submission, your investor pack and your " +
      "management information are the same numbers — because they're the same dataset.",
  },
  {
    id: "omnichannel",
    number: 4,
    title: "Three ways in",
    frames: 600,
    captions: [
      { at: 12, hold: 108, text: "Consume it however you already work." },
      {
        at: 132,
        hold: 168,
        text: "A managed service. Copilot. Or the workspace, when you need to drill in.",
      },
      { at: 312, hold: 108, text: "Same engine, same answer, every time." },
    ],
    narration:
      "Consume it however you already work. Run it as a managed service and never " +
      "log in. Ask it from Copilot, in the tools you have. Or open the workspace " +
      "when you need to drill in. Same engine, same answer, every time.",
  },
  {
    id: "close",
    number: 5,
    title: "Close",
    frames: 360,
    // The close is entirely burned-in display copy — the capability line and the ask
    // are the on-screen text, so a caption plate under them would only compete.
    captions: [],
    narration: "Trakt. A data operating system for specialist lenders.",
  },
];

/** Scene start frames, in order. */
export const sceneStarts = (): number[] => {
  let at = 0;
  return SCENES.map((scene) => {
    const start = at;
    at += scene.frames;
    return start;
  });
};

/** The absolute frame a scene starts on. */
export const startOf = (id: string): number => {
  const index = SCENES.findIndex((s) => s.id === id);
  if (index < 0) throw new Error(`Unknown scene ${id}`);
  return sceneStarts()[index];
};

export const totalFrames = (): number =>
  SCENES.reduce((total, scene) => total + scene.frames, 0);

export interface AbsoluteCaption extends Caption {
  scene: string;
  /** Absolute frame the caption appears on. */
  from: number;
}

/** Every caption, resolved to absolute frames. */
export const captions = (): AbsoluteCaption[] => {
  const starts = sceneStarts();
  return SCENES.flatMap((scene, i) =>
    scene.captions.map((c) => ({ ...c, scene: scene.id, from: starts[i] + c.at })),
  );
};

/** The still frames the outbound email body uses. */
export const STILL_FRAMES = [1020, 1500, 2100] as const;

export const timecode = (frame: number): string => {
  const total = frame / FPS;
  const mm = Math.floor(total / 60);
  const ss = Math.floor(total % 60);
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
};
