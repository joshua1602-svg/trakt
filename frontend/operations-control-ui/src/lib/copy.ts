/**
 * Every user-facing string used by the app chrome and labels.
 * Plain, calm, non-technical English only — a test enforces this.
 */
/**
 * How a reply was tied to a case, in words rather than in the correlator's own
 * vocabulary. An operator reads "the conversation it belongs to", not
 * "conversation" — and the distinction between evidence the mail system
 * carries and a mere coincidence of address is what they need to see.
 */
function basisLabel(basis: string): string {
  switch (basis) {
    case "conversation":
      return "the conversation it belongs to";
    case "in_reply_to":
      return "the message it replies to";
    case "case_ref":
      return "the reference in its subject";
    case "sender":
      return "the sender's address";
    default:
      return basis.replace(/_/g, " ");
  }
}

export const copy = {
  appName: "Trakt Operations",

  nav: {
    home: "Home",
    review: "Review",
    workflows: "Workflows",
    rules: "Rules",
    history: "History",
    agent: "OCC Agent",
    admin: "Platform configuration",
    onboarding: "Client onboarding",
    concentration: "Concentration tests",
    manual: "Manual delivery",
    specialistHeading: "Specialist",
    startNew: "Create a manual delivery",
    startAgent: "Start with OCC Agent",
  },

  onboarding: {
    title: "Client onboarding",
    subtitle:
      "Bring a new client into Trakt: ask the business questions, then create the configuration every later delivery is prepared from.",
    intro:
      "Start a new client and answer the questions as they come. Nothing is created until you approve and activate it at the end.",
    startNew: "Start new client onboarding",
    unavailable: "Client onboarding could not be loaded just now.",

    draftsHeading: "In progress",
    draftsDescription: "Onboarding you are still working on.",
    noDrafts: "Nothing in progress. Start a new client onboarding to begin.",
    awaitingHeading: "Waiting on the client",
    awaitingDescription: "Information has been asked for and has not come back yet.",
    noAwaiting: "Nothing is waiting on a client.",
    reviewHeading: "Ready for review",
    reviewDescription: "Everything answered, waiting for a decision.",
    noReview: "Nothing is waiting for review.",
    activeHeading: "Active clients",
    activeDescription: "Clients Trakt is configured for. Changes go through an amendment.",
    noActive: "No clients are active yet.",
    amend: "Amend",
    migrateHeading: "Bring in an existing client",
    migrateDescription:
      "Clients Trakt already serves whose configuration predates onboarding. This is optional.",
    migrate: "Bring in",

    caseTitle: "New client",
    back: "Back",
    next: "Next",
    migrationNote:
      "Trakt has filled in what it already holds for this client. Check every answer before approving, and resolve anything flagged.",

    reportingIntro:
      "Trakt works out what a client is eligible for from the books they have. You choose which of those they actually receive.",
    reportingHeading: "Reporting products",
    noRegimeFields:
      "The reporting products chosen so far need no standing regulatory information.",
    notHeldHere: "Not held here",

    sourcesIntro:
      "These follow from the portfolios you have added. Tell Trakt how each delivery arrives.",
    noSources: "Add a portfolio and Trakt will work out what it expects to receive.",
    addPipeline: "Also expect a pipeline book for",
    expectedLocation: "Deliveries expected at",

    checklistHeading: "What the client still needs to tell us",
    checklistDescription:
      "Worked out from the answers so far. Only questions the client can actually answer appear here.",
    checklistEmpty: "Nothing outstanding from the client.",
    requestSelected: "Ask the client for these",
    requestsHeading: "Requests raised",

    reviewIntro:
      "This is everything that will be created or changed. Nothing has been written yet.",
    stillOutstanding: "Information is still outstanding from the client.",
    willBeCreated: "What will be created",
    generatedHeading: "What Trakt has generated",
    generatedDescription: "Values Trakt mints. You do not type these.",
    defaultsHeading: "Defaults applied",
    whatChanges: "What changes",
    whatRecorded: "What will be recorded",
    unrepresentedHeading: "Recorded, but not written into configuration",
    unrepresentedDescription: "Kept with this client's record. Nothing today reads them.",
    approveHeading: "Approve",
    approveReason: "Why is this being approved?",
    reasonPlaceholder: "A short note for the record",
    approve: "Approve",
    approvedToast: "Approved. Activate it to create the configuration.",
    approvedNote:
      "Approved. Activating creates the client's configuration and registers what Trakt should expect to receive.",
    activateHeading: "Activate",
    activate: "Activate client",
    activatedToast: "Client activated.",

    withdraw: "Cancel this onboarding",
    withdrawHeading: "Cancel this onboarding?",
    withdrawExplain:
      "Nothing has been created for this client, so nothing is removed. The case is kept with your reason on it, and can be read afterwards.",
    withdrawExplainAmendment:
      "The configuration in force is untouched. This cancels the proposed change only.",
    withdrawReason: "Why is this being cancelled?",
    withdrawConfirm: "Cancel onboarding",
    withdrawnToast: "Onboarding cancelled. The record has been kept.",
    withdrawnHeading: "Cancelled",
    withdrawnNote:
      "This onboarding was cancelled and can no longer be edited. Nothing was created.",

    clientSubtitle: "The configuration in force for this client, and how it got there.",
    allClients: "All clients",
    identityHeading: "Identity",
    contactsHeading: "Contacts",
    presentationHeading: "Report presentation",
    sourcesHeading: "What Trakt expects to receive",
    sourcesDescription: "Created by onboarding, then kept up to date by real deliveries.",
    historyNote:
      "Every activation is kept. Earlier versions are never overwritten, so what was in force on any date can still be read.",
    whatChanged: "What changed",
    whatWasWritten: "What was written",
    casesHeading: "Onboarding and amendments",
    noCases: "No cases have been raised for this client.",
    tabs: {
      general: "General",
      entities: "Entities",
      portfolios: "Portfolios",
      reporting: "Reporting",
      sources: "Deliveries",
      history: "History",
      cases: "Cases",
    },
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
    heroHeading: "Tell Trakt what needs to happen",
    heroBody:
      "The OCC Agent will create the case, gather what is required, guide approvals, and " +
      "hand off to the governed workflow.",
    heroCta: "Start with OCC Agent",
    quickHeading: "Quick actions",
    quick: {
      onboardClient: "Onboard a new client",
      addPortfolio: "Add a portfolio",
      prepareDelivery: "Prepare a reporting delivery",
      reviewBlocked: "Review a blocked case",
      investigateFailed: "Investigate a failed run",
    },
    manualNote:
      "OCC Agent is the preferred route. Manual delivery remains available for specialist " +
      "and exceptional cases.",
    manualCta: "Create a manual delivery",
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
    title: "Create a manual delivery",
    intro:
      "Use this when files are not arriving through the normal automated intake process, " +
      "or when you need an ad hoc, backdated or replacement delivery.",
    stepLabel: "Step",
    outcomeHeading: "What should Trakt prepare?",
    outcomeMi: "MI Reporting",
    outcomeMiHelp: "The regular management information pack.",
    outcomeAnnex: "MI Reporting + ESMA Annex 2 delivery",
    outcomeAnnexHelp: "The regular pack plus the regulatory annex.",
    bookHeading: "Which book?",
    bookFunded: "Funded book",
    bookFundedHelp: "Loans already advanced. Regulatory reporting is prepared from this book.",
    bookPipeline: "Pipeline",
    bookPipelineHelp: "Cases not yet funded. Management information only — never a regulatory delivery.",
    bookPipelineLocksMi:
      "Pipeline is management information only, so the regulatory annex is not available for it.",
    detailsHeading: "Who is it for?",
    periodHeading: "Which reporting period?",
    clientLabel: "Client",
    clientPlaceholder: "Choose a client",
    newClientOption: "New client…",
    newClientLabel: "New client name",
    portfolioLabel: "Portfolio",
    periodLabel: "Reporting period",
    createButton: "Continue",
    filesHeading: "Upload files",
    uploadLabel: "Choose the files to send",
    uploadHelper:
      "Send spreadsheets or comma-separated files. Trakt decides where they are filed, " +
      "and only starts once the whole set has arrived safely.",
    chosenFiles: "Ready to send",
    noFilesChosen: "No files chosen yet.",
    destinationHeading: "Where Trakt will file this",
    confirmHeading: "Confirm and submit",
    uploadButton: "Send the files",
    uploading: "Sending…",
    removeFile: "Remove",
  },

  batch: {
    inputPack: "Input pack",
    roleReceived: "✓ Received and recognised",
    roleWaiting: "Waiting",
    roleOptional: "Optional — not provided",
    filesHeading: "Files received",
    noFiles: "No files have arrived yet.",
    configuration: "Configuration",
    configReady: "✓ Ready",
    configNeeded: "Configuration needed",
    blockingDecisions: "Blocking decisions",
    none: "None",
    addFilesHeading: "Add files",
    addFilesButton: "Add files",
    startButton: "Start onboarding",
    autoStartNote: "Trakt will start this on its own as soon as everything has arrived.",
    viewWorkflow: "See the workflow",
    notFound: "That input pack could not be found.",
  },

  workflows: {
    title: "Workflows",
    subtitle: "Everything Trakt is working on.",
    filterAll: "All",
    filterNeedsReview: "Needs review",
    filterBlocked: "Blocked",
    filterReady: "Ready to publish",
    filterPublished: "Published",
    filterCancelled: "Cancelled",
    empty: "No workflows here yet.",
    cancelledHidden: "Cancelled deliveries are kept but not listed here.",
  },

  workflow: {
    caseFile: "Delivery workflow",
    stepsHeading: "Where this delivery has got to",
    reviewDecisions: "Review decisions",
    runAgain: "Run again",
    approvePublish: "Approve and publish",
    hold: "Hold",
    publishConfirm: "Publish this report as the latest official version?",
    publishButton: "Publish",
    holdPrompt: "Why are you holding this report?",
    holdButton: "Hold this report",
    cancel: "Cancel this delivery",
    cancelHeading: "Cancel this delivery?",
    cancelExplain:
      "Nothing has been published, so nothing is withdrawn. The delivery is kept with your reason on it, and any questions still open on it stop being asked.",
    cancelExplainPublished:
      "This delivery has already been published. Cancelling does not withdraw what was published.",
    cancelPrompt: "Why is this being cancelled?",
    cancelButton: "Cancel delivery",
    cancelledToast: "Delivery cancelled. It has left the working list.",
    cancelledNote:
      "This delivery was cancelled. It is kept for the record and needs nothing from you.",
    warningsHeading: "Worth knowing",
    blockersHeading: "What's in the way",
    notFound: "That workflow could not be found.",
    filesHeading: "Files received",
    fileSize: "Size",
    fileArrived: "Arrived",
    fileCheck: "Version check",
    fileKind: "Recognised as",
    answered: "Answers already given",
    stillOpen: "Still to answer",
    answeredBy: "Answered by",
    outputsHeading: "What was produced",
    noOutputs: "No outputs were recorded.",
    openStep: "Open",
    collapseStep: "Hide",
    whatHappensNext: "What happens next",
    nextPending: "This step has not started yet.",
    nextNotApplicable: "This step does not apply to this delivery.",
    approveHeading: "Ready to publish",
    approveScopeHeading: "Should Trakt remember this decision for future deliveries?",
    approveConfirm: "Publish this delivery",
    evidenceHeading: "What Trakt is publishing",
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

  // The OCC Agent. The primary operating entry point: the operator describes
  // what needs to happen and Trakt creates and guides the case chronologically.
  agent: {
    title: "OCC Agent",
    subtitle: "Tell Trakt what needs to happen. It creates and guides the case, step by step.",
    syntheticChip: "Practice case",
    syntheticBanner:
      "Practice mode uses the real onboarding controls but does not activate configuration, " +
      "send email, or start the live pipeline.",
    newCase: "Start a new practice case",
    newCaseHeading: "What needs to happen?",
    newCasePrompt:
      "Tell Trakt in your own words. For example: “Onboard Northstar Lending. It is a UK " +
      "equity-release portfolio. They need monthly management information. The first files will " +
      "be a loan tape and executed cashflows.”",
    newCasePlaceholder: "Describe the client and what they need…",
    createButton: "Start",
    scenariosHeading: "Or start from a prepared example",
    scenarioRun: "Run this example",
    scenarioRunning: "Creating the practice case…",
    scenarioExpected: "Expected outcome",
    scenarioCreatedButNotOpened:
      "The practice case was created but could not be opened automatically. It appears in " +
      "the case list below.",
    openCase: "Open case",
    caseCreated: "Practice case created",
    casesHeading: "Practice cases",
    caseEmpty: "No practice cases yet.",
    filterAll: "All",
    filterNeedsYou: "Needs you",
    filterBlocked: "Blocked",
    filterReady: "Ready for execution",
    disabledTitle: "Not switched on",
    disabled:
      "The OCC Agent is not switched on for this environment. Ask your Trakt administrator.",

    // The route-level safety net: a failed action or a rendering fault shows a
    // card and keeps the rest of the OCC standing, never a blank page.
    errorTitle: "The OCC Agent hit a problem",
    errorBody:
      "The case and its record are safe. Try again, or open the case list. If this keeps " +
      "happening, tell your Trakt administrator.",
    errorRetry: "Try again",
    errorBack: "Back to the case list",

    // The compact case summary at the top of every case.
    summaryClient: "Client",
    summaryMode: "Mode",
    summaryModePractice: "Practice",
    summaryModeLive: "Live",
    summaryStage: "Current stage",
    summaryStreams: "Streams",
    streamPurpose: "Purpose",
    streamCadence: "Cadence",
    streamCadencePending: "pending confirmation",
    streamFile: "Required file",
    streamRegime: "Regime",
    streamRegimeNone: "Not applicable",
    streamRegimePotential: "Potentially applicable, subject to product selection",
    streamRegimeConfigured: "Configured",
    streamRegimeNotEligible: "Not eligible",
    streamsNone: "No streams declared yet. Tell Trakt what the client will provide.",

    // The chronological workflow.
    timelineHeading: "The onboarding, step by step",
    stageDone: "Done",
    stageCurrent: "Now",
    stageFuture: "Later",
    stageBlockedChip: "Blocked",
    stageLiveOnly: "Live onboarding only — practice stops before this",
    stageWhatRemains: "What remains",
    stageCompletedHeading: "What Trakt has completed",
    stageNoAction: "Nothing for you here right now — Trakt will move on on its own.",
    stageDetails: "Details",

    conversationHeading: "Conversation",
    conversationPlaceholder: "Tell Trakt what to do, or ask what is still needed…",
    send: "Send",
    sending: "Working…",
    proposalHeading: "Trakt is proposing a change",
    proposalConfirm: "Confirm and apply",
    proposalDismiss: "Not yet",

    statusHeading: "Where this case has got to",
    stageHeading: "Practice run",
    onboardingStageHeading: "Onboarding",
    onboardingHeading: "The onboarding",
    onboardingOpen: "Open it in the onboarding screens",
    checklistHeading: "What the client still has to tell us",
    checklistEmpty: "Nothing outstanding from the client.",
    checklistAsk: "Ask the client for these",
    checklistRecord: "Record what came back",
    requestsHeading: "Asked for",
    requestOutstanding: "Waiting on the client",
    requestAnswered: "Answered",
    previewHeading: "What activation would create",
    previewDescription:
      "The configuration this onboarding would generate. Nothing here has been created, and " +
      "this tab cannot create it.",
    previewNothingWritten: "Not created — practice case",
    previewNone: "There is not yet enough answered to generate a configuration.",
    factsHeading: "What the run is for",
    gatesHeading: "Controls",
    gateDone: "Passed",
    gateActive: "Now",
    gateBlocked: "Blocked",
    gatePending: "Not started",
    missingHeading: "Missing inputs",
    criteriaOnboarding: "The onboarding",
    criteriaExecution: "The practice run",
    criteriaBoundary: "The practice boundary",
    decisionsHeading: "Decisions waiting for you",
    artefactsHeading: "Files received",
    artefactIntended: "Where this would be filed",
    artefactNotWritten: "Not written — practice case",
    executionHeading: "Practice execution",
    readinessHeading: "Readiness",
    observationsHeading: "Worth knowing",
    blockersHeading: "What's in the way",
    packHeading: "Onboarding pack",
    packIssued: "Recorded as issued. No email was sent.",
    packDescription:
      "Everything Trakt needs from this client, drawn from the same governed catalogue the " +
      "onboarding screens use. Nothing here is a separate questionnaire.",
    packDraft: "Draft the pack",
    packRedraft: "Redraft it",
    packApprove: "Approve it for sending",
    packSend: "Record it as issued",
    packOutstanding: "still outstanding",
    packAnswered: "answered",
    packNone: "No pack has been drafted yet.",
    packDocument: "Show the document",
    packHide: "Hide it",
    packRecipients: "Goes to",
    packNoRecipient:
      "There is no contact address on this case yet. Record one, or type an address when you " +
      "issue it.",
    packMappingHeading: "About field mappings",
    packSteps: "Steps",
    packRequired: "required",
    packOptional: "optional",
    // The operator's own view of the client questions: readable, answerable
    // here on the client's behalf, and honest about what is NOT editable.
    questionsHeading: "The client questions",
    questionsShow: "Open them",
    questionsHide: "Close",
    questionsClosed:
      "Read every question this client is asked, and record answers here if you already " +
      "hold them.",
    questionsSave: "Save answers",
    questionsDiscard: "Discard changes",
    questionsSaved:
      "Saved. The answers are on the case, and any request that asked for them " +
      "closes once every item in it is answered.",
    questionsHint:
      "Answers are recorded exactly as typed, against the same keys a client's own " +
      "submission would use.",
    questionsPending: (n: number) =>
      n === 1 ? "1 answer not yet saved" : `${n} answers not yet saved`,
    questionsRequired: "Required",
    questionsOptional: "Optional",
    questionsUnanswered: "— not answered —",
    questionsLocked: "Not open yet, and will appear once the case says so:",
    questionsKnownHeading: "Already known — not asked again",
    questionsKnownHelp:
      "Trakt worked these out or was told them. To correct one, say so in the " +
      "conversation: the change is read, put to you, and recorded with who said it.",
    yes: "Yes",
    no: "No",
    mailHeading: "What the client has sent back",
    mailShow: "Check the mailbox",
    mailHide: "Close",
    mailClosed:
      "Look in the onboarding mailbox for replies to this case. Nothing is read or " +
      "changed until you ask.",
    mailChecking: "Looking…",
    mailRecheck: "Check again",
    mailNone: "Nothing has arrived for this case.",
    mailFrom: (who: string) => `From ${who}`,
    mailUnnamed: "an unnamed sender",
    mailMatchedOn: (bases: string[]) =>
      `Matched on ${bases.map(basisLabel).join(", ")}.`,
    mailTake: "Take this into the case",
    mailTaking: "Taking it in…",
    mailAlready: "Already taken in",
    mailCannotRegister:
      "This case is past the point where a file can be added to it. Take this reply in " +
      "from a case that is still gathering files, or say what it changes in the " +
      "conversation.",
    mailNoAttachments: "No attachments",
    mailOversize: "Too large for this deployment to read",
    mailUnreadable: "Trakt could not read this file",
    mailIngested: "Taken in. The files are on the case.",
    mailNotApplied:
      "The client's message is recorded on the case, not applied to it. Read it, then " +
      "tell the agent in the conversation if it should change an answer — the change is " +
      "put to you before anything is written.",
    mailUnmatchedHeading: "In the mailbox, but not tied to this case",
    mailUnmatchedHelp:
      "Trakt will not record a reply against a case unless the mail itself says which " +
      "one it is — the conversation it belongs to, the message it answers, or the " +
      "reference in its subject. A sender's address alone is not enough: one contact " +
      "can be on several onboardings. These need a person to look.",
    mailMailbox: (mailbox: string, folder: string) =>
      `Reading ${mailbox}${folder && folder !== "inbox" ? ` · ${folder}` : ""}`,

    packConfirmHeading: "Already known — check these are right",
    packConfirmNote:
      "Trakt worked these out or was told them. They are pre-populated, not asked again.",
    packNotAskedHeading: "What the client is NOT asked, and why",
    packLocked: "Not open yet",

    classificationHeading: "Who can answer what",
    classificationDescription:
      "Every field the governed catalogue declares, in one of five categories. Only the " +
      "second reaches a client.",
    classificationShow: "Show every field",
    classificationClientFacing: "asked of the client",
    packStatusHeading: "Where the pack has got to",
    packNotSent:
      "Trakt did not send this. Send the approved pack and covering email from the record.",

    reviewHeading: "Review and approval",
    reviewDescription:
      "The complete package a human approves: what Trakt holds, where every answer came from, " +
      "what is still outstanding, and exactly what activation would do.",
    reviewSubmit: "Submit for review",
    reviewShow: "Show the review package",
    reviewNone: "The case has not been submitted for review yet.",
    reviewApprove: "Approve the configuration",
    reviewApproveNote: "Approving records a decision. It starts nothing.",
    reviewProvenance: "Where it came from",
    reviewOperatorActions: "Actions for an administrator",
    reviewNotProvisioned: "Not provisioned",

    activationHeading: "Activation",
    activationDescription:
      "What confirming would do, in full. Approving the configuration is a separate act, and " +
      "does not start this.",
    activationConfirm: "Confirm and activate",
    activationConfirmLabel: "Type what you are confirming, for the record",
    activationRefusedHeading: "Why this cannot be activated",
    activationFiles: "Files that would be placed",
    activationTargets: "Where they would go",
    activationActions: "What would happen",
    activationDisabled:
      "Live execution is not switched on in this environment, so this is refused. The refusal " +
      "is recorded on the case.",
    activationStarted: "Ingestion started",

    disclosureUnderstood: "What Trakt understood",
    disclosureQuestions: "What Trakt needs to know",
    disclosureUnrecognised: "What Trakt could not read",
    disclosureNothingApplied:
      "Nothing has been applied. Confirm to apply only what is listed above, or tell Trakt the " +
      "rest.",
    auditHeading: "What happened",
    occLinksHeading: "Elsewhere in Trakt",
    nothingYet: "Nothing yet.",

    decisionIssue: "The issue",
    decisionEvidence: "Evidence",
    decisionRecommendation: "Trakt suggests",
    decisionConfidence: "Confidence",
    decisionMateriality: "Materiality",
    decisionConsequence: "If this is wrong",
    decisionActions: "Your answer",
    decisionApprove: "Accept",
    decisionReject: "Reject",
    decisionAnswered: "Answered",

    actionsHeading: "Governed controls",
    actionsHelp:
      "These are the same governed steps the conversation drives. Use whichever you prefer.",
    actionsInConversation:
      "What you can do next needs a detail Trakt has to be told — use the conversation:",
    actionsNone: "This case is finished. There is nothing further to do.",
    uploadHeading: "Provide the client response",
    uploadHelp:
      "Upload practice files, or use the files from a prepared example. They stay inside this " +
      "practice case.",
    uploadButton: "Add files",
    uploadFixture: "Use the example files",
    uploadGenerate: "Let Trakt make up a response",

    readyHeadline: "Practice case ready for execution.",
    readyNotDone: [
      "No live files were written.",
      "No production pipeline was triggered.",
      "No external email was sent.",
      "No client configuration was activated.",
      "Nothing was published.",
    ],
    readyStatus: "READY_FOR_EXECUTION",
    notReady: "Not ready yet",
    criteriaHeading: "Readiness criteria",
    manifestHeading: "Machine-readable summary",
    downloadPackage: "Show the readiness package",
    hidePackage: "Hide the readiness package",

    // The operator journey, in order. Labels only — where each stage starts
    // and ends is decided from the case's own states, never here.
    stages: {
      define: "Define onboarding",
      scope: "Confirm scope",
      pack_prepare: "Prepare onboarding pack",
      pack_review: "Review and approve the pack",
      pack_issue: "Record or issue the request",
      responses: "Receive client responses",
      artefacts: "Receive required artefacts",
      configure: "Generate configuration",
      config_review: "Review configuration",
      rehearsal: "Run rehearsal",
      exceptions: "Resolve exceptions",
      readiness: "Review readiness",
      approve_activation: "Approve activation",
      confirm_activation: "Confirm activation",
      ingestion: "Ingestion started",
    } as Record<string, string>,

    stageOutcomes: {
      deterministic_execution_completed: "Ran for real",
      contract_validation_completed: "Contract checked",
      execution_simulated: "Simulated only",
      human_input_required: "Needs you",
      hard_blocked: "Blocked",
    } as Record<string, string>,

    notFound: "That practice case could not be found.",
  },

  admin: {
    title: "Platform configuration",
    subtitle: "The settings administrators control, and how they are versioned.",
    tabs: {
      overview: "Overview",
      system: "System",
      assets: "Assets",
      regimes: "Regimes",
      history: "History",
    },
    accessDeniedTitle: "Access denied",
    accessDenied: "You do not have permission to administer platform configuration.",
    accessDeniedHelp: "If you think this is wrong, ask your Trakt administrator.",
    unavailable:
      "Configuration administration is temporarily unavailable. No configuration has been changed.",

    activeVersion: "Active version",
    draftVersion: "Draft version",
    priorVersions: "Earlier versions",
    status: "Status",
    activatedAt: "Activated",
    activatedBy: "Activated by",
    createdBy: "Created by",
    createdAt: "Created",
    basedOn: "Based on version",
    fingerprint: "Fingerprint",
    partCount: "Parts",
    notes: "Notes",
    noNotes: "No notes were added.",

    sections: {
      summary: "Summary",
      dependencies: "Dependencies",
      files: "Parts",
      history: "History",
      impact: "Impact",
      compare: "Compare",
    },

    actions: {
      reviewActive: "Review active version",
      createDraft: "Create a draft",
      compare: "Compare",
      validate: "Check this draft",
      activate: "Activate",
      rollback: "Roll back",
      inspect: "Inspect",
      viewPart: "Open",
      closePart: "Close",
      showTechnical: "Show technical details",
      hideTechnical: "Hide technical details",
    },

    draft: {
      heading: "Draft awaiting action",
      none: "There is no draft version for this package.",
      chip: "Draft",
      changedParts: "What changed",
      noChanges: "This draft is an exact copy of the version it was based on.",
      createPrompt: "Create a draft copy of the active version?",
      createHelp:
        "A draft changes nothing on its own. It must pass its checks and be activated before Trakt uses it.",
      createButton: "Create draft",
      created: "Draft created. Check it before activating.",
      readOnlyNote:
        "Drafts are reviewed here and prepared through the supported change process. This screen never edits an active version.",
    },

    validation: {
      heading: "Checks",
      notChecked: "Not checked yet",
      pass: "Checks passed",
      fail: "Validation failed",
      running: "Checking…",
      done: "Checks finished.",
      problemsHeading: "What is in the way",
      warningsHeading: "Worth knowing",
      dependenciesHeading: "What this affects",
    },

    activate: {
      heading: "Activate this version?",
      package: "Package",
      version: "Version",
      replaces: "Replaces version",
      pinnedNote: "Workflows that have already started stay on the version they began with.",
      futureNote: "Only workflows started after this point will use the new version.",
      historyNote: "Reports that have already been published are unchanged.",
      button: "Activate",
      done: "Activated. Future workflows will use this version.",
      blockedHeading: "This cannot be activated",
      invalidHelp: "This version has not passed its checks yet.",
    },

    rollback: {
      heading: "Roll back to an earlier version?",
      help: "The version being replaced is kept, not deleted. You can move forward again later.",
      button: "Roll back",
      done: "Rolled back. Future workflows will use the earlier version.",
      choose: "Which version should Trakt go back to?",
    },

    compare: {
      heading: "Compare versions",
      from: "From",
      to: "To",
      added: "Added",
      changed: "Changed",
      removed: "Removed",
      unchanged: "Unchanged",
      identical: "These two versions are identical.",
      truncated: "Only the first part of this change is shown.",
      pick: "Choose two versions to compare.",
    },

    files: {
      heading: "What is in this package",
      empty: "This version has no parts.",
      size: "Size",
      readOnly: "Read only. Active versions can never be edited here.",
    },

    compatibility: {
      heading: "Which assets can report under which regimes",
      supported: "Supported",
      notSupported: "Not supported",
      asset: "Asset",
      blockerHeading: "Compatibility",
      none: "No compatibility problems were found.",
    },

    impact: {
      heading: "Potential impact",
      clients: "Clients using this package",
      portfolios: "Portfolios using this package",
      running: "Running workflows pinned to this version",
      pinned: "Workflows pinned to this version",
      future: "Future workflows affected after activation",
      yes: "Yes",
      workflowsHeading: "Pinned workflows",
      none: "Nothing is using this version yet.",
    },

    audit: {
      title: "Configuration history",
      subtitle: "Every configuration change, who made it, and what it means.",
      empty: "No configuration changes have been made yet.",
      chainIntact: "This history is complete and unaltered.",
      chainBroken: "This history could not be confirmed as complete. Tell your administrator.",
    },

    overview: {
      heading: "Platform configuration",
      systemHeading: "System package",
      assetHeading: "Asset packages",
      regimeHeading: "Regime packages",
      attentionHeading: "Needs an administrator",
      noAttention: "Nothing needs an administrator right now.",
      draftsAwaiting: "Drafts awaiting action",
      validationFailures: "Checks that failed",
      requiresReview: "Not checked yet",
      compatibilityIssues: "Worth knowing",
      recentActivations: "Recent activations",
      recentRollbacks: "Recent roll backs",
      noRecent: "No recent changes.",
    },

    system: {
      title: "System configuration",
      subtitle: "The platform-wide settings every client shares.",
    },

    // The operator-first view of an asset package. The administrative and
    // technical detail is still here — it is one disclosure away, not gone.
    readiness: {
      heading: "Can this configuration safely process a delivery?",
      state: "Readiness",
      warnings: "Worth knowing before you rely on this",
      noWarnings: "Nothing to flag.",
      technicalHeading: "Administration and technical details",
      show: "Show administration and technical details",
      hide: "Hide administration and technical details",
      configurationSuffix: "configuration",
      identifier: "Package identifier",
    },

    assets: {
      title: "Asset configuration",
      subtitle: "The product settings that describe how each asset type behaves.",
      supportedRegimes: "Supported regimes",
      sourceSemantics: "How source data is read",
      mappingDefaults: "Default settings",
      taxonomies: "Groupings",
      issuePolicy: "Issue handling rules",
      profiles: "Product profiles",
      settings: "Settings",
      empty: "No asset packages are configured.",
      clientNote:
        "Client and portfolio settings are not administered here. They stay part of onboarding.",
    },

    regimes: {
      title: "Regime configuration",
      subtitle: "The regulatory packages Trakt can report under.",
      regulator: "Regulator",
      annex: "Annex",
      packageVersion: "Package version",
      schema: "Schema",
      draftSchema: "This is a draft schema published by the regulator.",
      fieldUniverse: "Fields covered",
      codeOrder: "Field order",
      validationPolicy: "Checking policy",
      noDataPolicy: "Missing-value policy",
      unknownValues: "Unexpected values",
      deferred: "Deferred fields",
      allowedCodes: "Codes allowed when a value is missing",
      fieldsAllowing: "Fields that may be left empty",
      compatibleAssets: "Compatible assets",
      empty: "No regime packages are configured.",
    },
  },

  concentration: {
    title: "Concentration tests",
    subtitle:
      "Review what the client supplied, decide each proposed test, and activate "
      + "the approved set as the next governed version.",
    pickClient: "Choose a client to review their concentration tests.",
    open: "Open review",
    extractHeading: "Client response",
    extractHelp:
      "Paste the client's covenant wording or limits table exactly as supplied. "
      + "Trakt reads it deterministically; anything unclear becomes a question, "
      + "never a guess.",
    sourceReference: "Where this wording came from",
    responseText: "Response wording",
    extract: "Read the response",
    noResponse:
      "No client response has been read yet. Paste the covenant wording or "
      + "limits table above — or record the onboarding outcome as not "
      + "applicable if this client has no concentration tests.",
    noneFound:
      "Nothing extractable was found in that wording. Tests need a comparison "
      + "and a number; ask the client for the schedule if this looks wrong.",
    proposalsHeading: "Proposed tests",
    filterAll: "All statuses",
    detailHeading: "Proposal detail",
    selectPrompt: "Select a proposal to review its wording, mapping and concerns.",
    sourceWording: "Source wording",
    proposedMetric: "Proposed measure",
    matchOutcome: "Match",
    parameters: "Parameters",
    threshold: "Threshold",
    operator: "Direction",
    confidence: "Confidence",
    concerns: "Concerns",
    questions: "Confirmation questions",
    answerLabel: "Answer",
    recordAnswer: "Record answer",
    editHeading: "Edit permitted fields",
    parametersJson: "Parameters (JSON)",
    effectiveDate: "Effective date",
    saveEdits: "Save edits",
    approvalBlockedHeading: "Approval is blocked",
    approve: "Approve",
    approveComments: "Approval comments",
    reject: "Reject",
    unsupported: "Mark unsupported",
    notApplicable: "Not applicable",
    clarify: "Ask the client",
    clarifyQuestion: "What should the client be asked?",
    supersede: "Supersede",
    reasonLabel: "Reason",
    activationHeading: "Activation",
    activationIntro:
      "Activating creates the next governed version from every approved "
      + "proposal. Nothing else is included.",
    willActivate: "These tests will become active:",
    nothingToActivate:
      "Nothing is approved yet, so there is nothing to activate.",
    activate: "Activate approved tests",
    currentVersion: "Active version",
    versionsHeading: "Version history",
    auditHeading: "Decision history",
    invalidJson: "The parameters are not valid JSON.",
    extractionFailed:
      "The response could not be read. Nothing was saved; the wording is "
      + "unchanged.",
    permissionNote: "Approval, supersede and activation need an administrator.",
  },

  statusLabels: {
    pending_confirmation: "Pending confirmation",
    pending_approval: "Pending approval",
    clarification_requested: "Clarification requested",
    proposed: "Proposed",
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
    receiving: "Receiving files",
    incomplete: "Waiting for files",
    classifying: "Looking at the files",
    review_required: "Needs review",
    configuration_required: "Configuration needed",
    // Delivery workflow steps.
    complete: "Complete",
    current: "Current",
    pending: "Pending",
    not_applicable: "Not applicable",
    // Configuration package lifecycle (the administrator area).
    draft: "Draft",
    active: "Active",
    superseded: "Replaced",
    valid: "Valid",
    not_checked: "Not checked",
    passed: "Valid",
    failed_checks: "Not valid",
  } as Record<string, string>,
};

export function decisionsLabel(n: number): string {
  return `${n} ${n === 1 ? copy.common.decisionOne : copy.common.decisionMany}`;
}
