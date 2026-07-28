/**
 * Every user-facing string used by the app chrome and labels.
 * Plain, calm, non-technical English only — a test enforces this.
 */
export const copy = {
  appName: "Trakt Operations",

  nav: {
    home: "Home",
    review: "Review",
    workflows: "Workflows",
    rules: "Rules",
    history: "History",
    startNew: "Start something new",
  },

  signIn: {
    title: "Trakt Operations",
    prompt: "Paste your access key",
    helper: "Your access key was shared with you by your Trakt administrator.",
    placeholder: "Access key",
    button: "Continue",
  },

  errors: {
    network: "Trakt could not be reached. Check your connection and try again.",
    signedOut: "Your access key is no longer valid. Please sign in again.",
    generic: "Something went wrong. Please try again.",
    retry: "Try again",
  },

  common: {
    loading: "Loading…",
    decisionOne: "decision",
    decisionMany: "decisions",
    version: "Version",
    cancel: "Cancel",
    confirm: "Confirm",
    save: "Save",
    close: "Close",
    optionalReason: "Why? (optional)",
    showDetails: "Show details",
  },

  home: {
    title: "Trakt Operations",
    subtitle: "What Trakt is preparing for you, and what needs you.",
    tiles: {
      new_deliveries: "New deliveries",
      needs_attention: "Needs your attention",
      blocked: "Blocked",
      ready_to_publish: "Ready to publish",
      recently_published: "Recently published",
    },
    needsAttention: "Needs your attention",
    recentlyPublished: "Recently published",
    emptyAttention: "Nothing needs your attention.",
    emptyPublished: "Nothing has been published recently.",
  },

  newWorkflow: {
    title: "Start something new",
    outcomeHeading: "What would you like Trakt to prepare?",
    outcomeMi: "MI Reporting",
    outcomeMiHelp: "The regular management information pack.",
    outcomeAnnex: "MI Reporting + ESMA Annex 2",
    outcomeAnnexHelp: "The regular pack plus the regulatory annex.",
    detailsHeading: "Who is this for?",
    clientLabel: "Client",
    clientPlaceholder: "Choose a client",
    newClientOption: "New client…",
    newClientLabel: "New client name",
    portfolioLabel: "Portfolio",
    periodLabel: "Reporting period",
    filesHeading: "Which files should Trakt use?",
    pickDelivery: "Choose a recent delivery",
    noDeliveries: "No recent deliveries for this client yet.",
    orRegister: "Or point Trakt at new files",
    folderLabel: "Where are the files?",
    folderHelper: "Paste the folder location your files were saved to.",
    registerButton: "Check the files",
    classificationHeading: "Here's what Trakt thinks",
    changeLink: "Change",
    typeNewClient: "New client onboarding",
    typeNewPortfolio: "New portfolio onboarding",
    typeRecurring: "Recurring reporting",
    typeBackfill: "Historical backfill",
    startButton: "Start",
    filesReceived: "files received",
  },

  workflows: {
    title: "Workflows",
    subtitle: "Everything Trakt is working on.",
    filterAll: "All",
    filterNeedsReview: "Needs review",
    filterBlocked: "Blocked",
    filterReady: "Ready to publish",
    filterPublished: "Published",
    empty: "No workflows here yet.",
  },

  workflow: {
    reviewDecisions: "Review decisions",
    runAgain: "Run again",
    approvePublish: "Approve and publish",
    hold: "Hold",
    publishConfirm: "Publish this report as the latest official version?",
    publishButton: "Publish",
    holdPrompt: "Why are you holding this report?",
    holdButton: "Hold this report",
    warningsHeading: "Worth knowing",
    blockersHeading: "What's in the way",
    notFound: "That workflow could not be found.",
  },

  reviews: {
    title: "Review",
    subtitle: "Questions Trakt needs you to answer.",
    empty: "Nothing to review right now.",
    blockingChip: "Needed to continue",
    optionalChip: "Optional",
    recommendedHeading: "Trakt suggests",
    scopeHeading: "Where should this answer apply?",
    somethingElse: "It's something else",
    somethingElsePlaceholder: "Tell Trakt what this means",
    confirm: "Confirm",
    reject: "This is wrong",
    rejectReason: "Tell Trakt why, so it can do better",
    savedRerun: "Saved. Trakt is re-running the affected step.",
    saved: "Saved.",
    backToList: "Back to review",
    notFound: "That question could not be found.",
  },

  scopes: {
    file: "This delivery only",
    portfolio: "This portfolio",
    client: "This client",
    global: "All of Trakt",
  },

  rules: {
    title: "Rules",
    subtitle: "Everything Trakt has learned, approved by your team.",
    searchPlaceholder: "Search rules",
    kindAll: "All kinds",
    scopeAll: "All scopes",
    sourceTerm: "What the file says",
    approvedMeaning: "What it means",
    approvedBy: "Approved by",
    historyHeading: "Earlier versions",
    empty: "No rules match.",
  },

  history: {
    title: "History",
    subtitle: "Every report Trakt has prepared, by client.",
    builtWithPrefix: "built with",
    builtWithSuffix: "approved rules",
    previousVersion: "Previous version available",
    empty: "No reports yet.",
  },

  statusLabels: {
    received: "Received",
    running: "In progress",
    needs_review: "Needs review",
    blocked: "Blocked",
    awaiting_publication: "Ready to publish",
    published: "Published",
    held: "On hold",
    cancelled: "Cancelled",
    failed: "Did not finish",
    waiting: "Waiting",
    ready: "Ready",
    approved: "Approved",
    rejected: "Rejected",
    completed: "Done",
    open: "Open",
    resolved: "Resolved",
    prepared: "Prepared",
  } as Record<string, string>,
};

export function decisionsLabel(n: number): string {
  return `${n} ${n === 1 ? copy.common.decisionOne : copy.common.decisionMany}`;
}
