(function () {
  "use strict";

  var OLLAMA = "http://localhost:11434";
  var DEFAULT_MODEL = "qwen3:14b";
  var PREFERRED_EMBED = "nomic-embed-text";
  var INDEX_URL = "/assistant_index.json";
  // Corpus vectors precomputed at build time (tools/build_assistant_vectors.js); saves the first-visit embed wait.
  var VECTORS_URL = "/assistant_vectors.json";
  var MIN_MODEL_PARAMS_B = 4.0;
  var TOP_K = 6;
  var HISTORY_TURNS = 10;
  // 0 means NO cap: capText returns the text unchanged when cap <= 0, so evidence chunks reach the model whole.
  var EVIDENCE_CHAR_CAP = 0;
  // Weight on the semantic (cosine) side; BM25 gets 1 - alpha. Both arrays are min-max normalised first.
  var HYBRID_ALPHA = 0.5;
  // Caps only the copy kept in history; what the user sees is never truncated.
  var HISTORY_ANSWER_CHAR_CAP = 500;
  // Restored by setIndexing, so it lives here rather than inline in the panel markup alone.
  var INPUT_PLACEHOLDER = "Ask about GEM-pRF…";
  // chatComplete only -- the stream path never retries. STREAM_IDLE_MS gaps chunks after a first LLM_TIMEOUT_MS window.
  var LLM_TIMEOUT_MS = 180000;
  var LLM_RETRIES = 3;
  var STREAM_IDLE_MS = 90000;
  // Substring probes against the installed model name; the prefixes are baked into every cached vector.
  var EMBED_PREFIX = {
    "nomic-embed-text": { doc: "search_document: ", query: "search_query: " },
    "mxbai-embed-large": { doc: "", query: "Represent this sentence for searching relevant passages: " },
    "e5": { doc: "passage: ", query: "query: " },
    "bge": { doc: "", query: "Represent this sentence for searching relevant passages: " }
  };

  // The one axis a value choice can turn on. Five prompts ask for it; they must offer the SAME three.
  var GOAL_TRIAD = "accuracy, runtime, or GPU memory";

  // Grounded-answer prompt: streamChat and generateAnswer only. Other branches have their own *_SYSTEM.
  var SYSTEM_PROMPT =
    "You answer GEM-pRF questions using only the provided evidence. " +
    "Do not use outside knowledge — including textbook, training-data, or general " +
    "domain facts that the evidence itself does not state. This applies to " +
    "rationales, mechanisms, and 'why' explanations as much as to numbers and names.\n\n" +
    "Decision rule:\n" +
    "  - If ANY evidence item contains the requested fact (verbatim, paraphrased, or " +
    "split across items), you MUST answer with that fact. Do not refuse, do not hedge.\n" +
    "  - If an evidence item explicitly states that the source does NOT contain a " +
    "particular fact (e.g., 'the paper does not give a rationale', 'the corpus does " +
    "not contain a deeper definition'), RESPECT that disclaimer: report only what the " +
    "evidence does say, and explicitly note that the source does not address the rest. " +
    "Do not paper over the gap with a citation that does not actually support the claim.\n" +
    "  - The user's question may use vague or anaphoric phrasing (\"it\", \"result\", " +
    "\"how does it work\"). Resolve the reference from the CONVERSATION HISTORY block, then " +
    "answer the resolved question from the evidence.\n" +
    "  - Code is documentation for STRUCTURE. When evidence includes code (function " +
    "bodies, methods, tests), treat the code as authoritative and INFER the structure " +
    "of values, files, and APIs from how the code constructs or consumes them. Example: " +
    "if a serializer writes `json_entry = args2jsonEntry(muX, muY, sigma, r2, ...)`, the " +
    "JSON contains those fields. State the inference plainly, grounding it in that `code` item. " +
    "This license applies to code structure only — not to broader rationales, theory, or " +
    "user-facing semantics.\n" +
    "  - When no evidence item contains or supports the requested fact at all, your ENTIRE reply " +
    "must be 'INSUFFICIENT_EVIDENCE: <one-sentence reason>', starting with that token. Never write " +
    "the token INSUFFICIENT_EVIDENCE anywhere else, in any other sentence.\n\n" +
    "Source-type discipline (CRITICAL for refusal correctness):\n" +
    "  Every evidence item is tagged with its class — `prose`, `sample`, or `code`. Trust the " +
    "tag; do not infer type from the source_id.\n" +
    "    • `prose` — the paper, the website, and project READMEs. Authoritative for behavior, " +
    "recommendations, UI semantics, defaults, and rationale.\n" +
    "    • `sample` — attribute values read out of example XML config files. Illustrative sample " +
    "data, NOT normative defaults or recommendations.\n" +
    "    • `code` — source and tests. Authoritative for code structure and what fields/keys " +
    "exist; not for UI behavior or recommended settings.\n" +
    "  For three question classes ONLY — (a) UI behavior (\"what happens when I check/enable/" +
    "select X\"), (b) recommended or default values (\"what is the default Y\", \"what does " +
    "GEM-pRF recommend\"), and (c) runtime semantics of a configurator toggle — you MUST ground the " +
    "answer in a `prose` item that literally states the behavior, and this OVERRIDES the general 'if ANY " +
    "evidence contains the fact, answer' rule above. A `sample` attribute value does not " +
    "establish a default or a recommendation; a `code` if/else branch does not establish UI " +
    "semantics. With no `prose` item stating the behavior, refuse with 'INSUFFICIENT_EVIDENCE: " +
    "<reason>' rather than synthesizing from sample values or implementation details.\n" +
    "  For every OTHER question class — definitions, citations, what fields/keys exist, code " +
    "structure, numbers stated in prose — the general rule stands and `sample`/`code` items ARE " +
    "valid grounding; do not refuse those because of their class.\n\n" +
    "Endorsement discipline: A documented default or a value stated in the sources is a FACT " +
    "about what the software does, not an endorsement. Report defaults and stated values plainly " +
    "('the default is X', 'the sample config sets X'). Do NOT assert that a value is 'appropriate', " +
    "'correct', 'the right choice', 'safe to use', 'suitable', or 'recommended' for the user's case " +
    "unless a `prose` item explicitly recommends it, and NEVER manufacture " +
    "certainty ('we can be certain', 'this is definitely appropriate') the evidence does not state.\n\n" +
    "Completeness rule: Include every factual detail the question implicitly asks for. " +
    "Quote numbers, dates, version IDs, software/container names, DOIs, journal volumes, " +
    "and article numbers verbatim from the evidence. If the question asks 'in which " +
    "journal and year', include the journal name AND the year AND the volume AND the " +
    "article identifier when present. If the question asks 'which Docker container', " +
    "include the exact container name.\n\n" +
    "Style rule: Be concise but exact. Prefer short, factual sentences. Write in plain " +
    "English. Never reproduce an item's bracketed prefix (e.g. `[prose | paper.full ...]`) or a bare " +
    "source id as a citation in the answer text — provenance is surfaced separately by the caller via " +
    "the retrieved evidence list. The heading inside that prefix IS evidence: a fact carried only there " +
    "(a title, a journal, a year, a version) is yours to use — state it in your own sentence rather than " +
    "quoting the label around it. Numerical values, defaults, ranges, and " +
    "mathematical formulas in plain notation are fine; quoted strings from log messages are " +
    "fine.\n" +
    "  Code is first-class evidence, exactly like the paper and the docs — not something to " +
    "be hidden. When the question asks about the source, implementation, or the exact " +
    "code/lines/logic, AND a `code` evidence item contains it, quote the relevant lines " +
    "verbatim in a fenced code block and then explain in plain English what they do. Quote " +
    "only what a code chunk literally contains; never reconstruct or invent code that is not " +
    "in the evidence.\n" +
    "  For every OTHER question class (definitions, behaviour, recommendations, 'why' " +
    "explanations), answer in plain English and do NOT sprinkle raw identifiers into the " +
    "prose: refer to configurator fields by their UI labels (e.g. 'Normalize HRF', 'Default " +
    "GPU', 'Refine Fitting') rather than their XML paths (e.g. /root/stimulus/binarization/" +
    "@enable), module or function names (e.g. gem.init_setup.manage_gpus, np.linspace), " +
    "internal variable names (e.g. batch_size, total_y_signals, num_frames_downsampled), or " +
    "environment variables (e.g. os.environ['CUDA_VISIBLE_DEVICES']). Say what the code does " +
    "in the user's terms -- 'the run divides the signals by this value to size each batch', " +
    "not 'batch_size = total_y_signals / num_batches'.\n" +
    "  Much of the prose evidence is itself written in that internal register, naming XML " +
    "paths, classes, and variables inline. That is how the source documents it, NOT a licence " +
    "to answer that way: translate it, and never let a chunk's phrasing carry identifiers into " +
    "your prose just because they sit in the sentence you are drawing the fact from. The " +
    "distinction is purpose, not source type: quote code when the user wants the code, " +
    "describe it in plain English when the user wants the concept.\n\n" +
    "Do not invent numbers, citations, authors, container names, version IDs, " +
    "affiliations, rationales, mechanisms, or explanations that are not present in the " +
    "evidence, and never offer a value as one the field commonly uses.\n\n" +
    "A CONVERSATION HISTORY block may precede the question. Use it ONLY to resolve what " +
    "the question refers to (e.g. 'it', 'that parameter', a short follow-up). Prior " +
    "answers are NOT evidence: every factual claim must still be grounded in the " +
    "evidence items below, and the refusal rules apply to the resolved question.";

  var INSUFFICIENT_EVIDENCE_MESSAGE =
    "I do not have enough support in the allowed GEM-pRF sources to answer that reliably. " +
    "This prototype is restricted to the paper, GEM-pRF docs, and the published package code.";

  // Refusal contract: anchored at the reply's start like CLARIFY:, surviving a leading quote or **.
  function isRefusal(text) {
    return /^["'`*\s]*INSUFFICIENT_EVIDENCE\b/i.test(String(text || ""));
  }

  // Human-prompt suffix when clarifying is allowed; the model replies with one CLARIFY: line.
  var CLARIFY_DIRECTIVE =
    "\n\nYou may, for THIS answer only, ask ONE clarifying question -- but ONLY in these cases, and only " +
    "when a brief answer from the user would genuinely change what you say:\n" +
    "- They are asking which value or setting to CHOOSE or USE for their own analysis (a decision, not a " +
    "definition) and have not given the goal, data, or constraint that would determine it: ask about " +
    "that. Do not restate what the setting does, do not answer your own question, and do not ask them to " +
    "name the value -- that is exactly what they are asking you.\n" +
    "- The subject of the question could refer to two or more DISTINCT settings and you cannot tell which " +
    "is meant: ask which one, instead of refusing or silently picking one.\n" +
    "- The question names no specific setting at all: ask which they mean.\n" +
    "For any of these, reply with exactly one line and nothing else: CLARIFY: <the one short question>.\n" +
    "In every OTHER case, answer or refuse normally -- do not ask a clarifying question. A question about " +
    "what something IS, MEANS, or DOES is a definition: answer it from the evidence. Once the user has " +
    "given a goal (now or earlier), answer too, reasoning from the documented mechanism toward it and " +
    "saying plainly when the documentation does not establish how the value affects that goal.";

  // Human-prompt suffix, set only on the turn that ANSWERS a value-choice clarify -- never on the first ask.
  var VALUE_ANSWER_DIRECTIVE =
    "\n\nThe user is choosing a value for a setting and has now given their goal or constraint. Using ONLY " +
    "the evidence, write a short natural-prose answer (no headings, no numbered list). The system rules " +
    "above still bind -- in particular, do not endorse a value and do not invent one.\n" +
    "Answer the input they actually gave. Name it back to them and say what it does and does not settle -- " +
    "an answer that never mentions what they just told you has ignored them. First, explain the governing " +
    "logic from the evidence -- what this value controls and what a good choice depends on (another " +
    "setting, the stimulus, the dataset, or the hardware) -- reasoning toward what they gave. Where the " +
    "evidence states a CONSTRAINT their input runs into (something that must fit, a limit that is exceeded, " +
    "or a different setting that actually governs their case), say so and name that setting: a documented " +
    "constraint answers them as squarely as a number does.\n" +
    "Then state every concrete value the evidence carries for this setting, INCLUDING one that appears only " +
    "in a sample or default configuration: name it and say what it is (a documented default, not a " +
    "recommendation) rather than calling the evidence silent. Say the evidence documents no value only when " +
    "it shows none. Where the evidence gives a RULE instead of a number (e.g. 'x times the stimulus'), state " +
    "the rule AS a rule and never supply operands of your own to illustrate it.\n" +
    "Land it. When the evidence carries a documented value AND a direction to move it, put the two together " +
    "for them -- the documented value is where the documentation starts them, the documented direction is " +
    "how it says to move from there -- attributing both to the documentation. When the evidence cannot " +
    "convert what they gave into a number, say that plainly in one sentence and then give them what it DOES " +
    "support, rather than closing on what is missing. Do NOT end with a question.";

  // Used when clarifying is allowed; the flat refusal is used when it is not.
  var INTERACTIVE_NO_EVIDENCE_MESSAGE =
    "I don't have a grounded answer for that in the GEM-pRF paper, docs, or package code. If you're " +
    "configuring a setting, tell me which one and what you're optimizing for -- " + GOAL_TRIAD +
    " -- and I'll answer from the documentation.";

  var NO_EVIDENCE_TURN_SYSTEM =
    "You are the GEM-pRF configuration assistant. GEM-pRF configures population receptive field (pRF) " +
    "analyses. The user's latest message retrieved NO supporting documentation, so you have no evidence " +
    "in front of you. Reply in one or two sentences:\n" +
    "  - If it is conversational or meta (a greeting, thanks, small talk, or asking what you need from " +
    "them), respond naturally and briefly, then steer them toward their configuration: invite them to " +
    "name the setting they're configuring and what they're optimizing for (" + GOAL_TRIAD + ").\n" +
    "  - If it is a factual GEM-pRF question you have no documentation for, say plainly that the GEM-pRF " +
    "sources (paper, docs, package code) do not cover it. Do not guess.\n" +
    "Hard rule: you have NO evidence, so state NO GEM-pRF fact, default, parameter value, formula, or " +
    "behavior, and never recall specifics from outside the provided sources. When unsure whether " +
    "something is conversational or factual, decline and redirect rather than assert.";

  var NEEDS_GOAL_SYSTEM =
    "You triage questions for a GEM-pRF configuration assistant. Some settings have a best value that " +
    "depends on the user's goal (" + GOAL_TRIAD + ").\n" +
    "Answer YES only if the user is asking which value to CHOOSE or USE for a setting AND the message " +
    "gives NO purpose, goal, or constraint to guide that choice.\n" +
    "Answer NO in every other case, including:\n" +
    "- a question about what something IS, MEANS, or DOES (a definition or fact);\n" +
    "- a request that already states ANY purpose, goal, or constraint -- e.g. 'for high accuracy', 'to " +
    "cover my stimulus', 'so I can ...', 'fastest', 'to save memory'. A stated purpose means the answer " +
    "should reason toward it, not ask for it.\n" +
    "Reply with exactly one word: YES or NO.";

  var VALUE_CLARIFY_SYSTEM =
    "The user is choosing a value for a GEM-pRF setting but has not given the information needed to " +
    "recommend one. Ask ONE short, natural clarifying question for the input that would let the " +
    "documentation below actually settle the value.\n" +
    "Their reply is answered from that documentation ALONE -- no outside formula, convention, or estimate. " +
    "So before you ask, test your question: if the user answered it, could the documentation below turn " +
    "that answer into THIS setting's value? Only ask if it can. Which case you are in depends on what the " +
    "documentation carries:\n" +
    "- It gives a rule that SETS this setting from a quantity (e.g. 'this value should be x times the " +
    "stimulus radius'): ask for that quantity in the real-world form the user knows it -- the scan/run " +
    "length (or the TR and number of volumes), the stimulus's extent on screen. Prefer that operational " +
    "input over an abstract internal quantity: for a low-frequency-drift regressor, ask how long their runs " +
    "are and what high-pass cutoff they want, NOT 'how much drift do you expect'.\n" +
    "- Careful: a formula that computes something ELSE from this setting does not run backwards. If the " +
    "documentation says a batch size is the signal count divided by this setting, the signal count does not " +
    "pick the setting -- it only says what a chosen value would work out to. Asking for it buys nothing. " +
    "The same holds for any quantity the documentation merely mentions near the setting without using it to " +
    "determine the value.\n" +
    "- Otherwise the documentation gives only a DIRECTION (larger values cost less peak memory but more " +
    "iterations) and no way to compute a number: ask which way they want to trade -- " + GOAL_TRIAD +
    ". A direction can serve a goal, but it cannot turn a hardware figure or a dataset size into " +
    "a value, so do NOT ask for a number the documentation has no way to spend. A question whose answer " +
    "cannot change the reply is worse than no question.\n" +
    "You may use general knowledge to judge WHICH input the documentation needs, but you must NOT assert " +
    "any GEM-pRF fact, default, formula, or value -- only ask the question. You may add that they can " +
    "say 'just the default' for the documented value.\n" +
    "Do NOT repeat or rephrase the user's own question back to them. Ask about ONLY the setting the user " +
    "asked about; never introduce or offer a DIFFERENT parameter as an alternative. Reply with only the " +
    "one question, nothing else.";

  var CONDENSE_SYSTEM =
    "You rewrite a user's latest question into a STANDALONE GEM-pRF question. If it refers to " +
    "something from the conversation ('it', 'that parameter', 'what value in my case?'), fold that " +
    "referent in so the question stands on its own. If it is already self-contained, return it " +
    "UNCHANGED -- a question that names its own subject (e.g. 'What is nDCT?', 'What does " +
    "binarization do?') is self-contained even when earlier turns discussed other settings, so do " +
    "NOT graft that prior context onto it. Whatever the latest question leaves unsaid comes from the " +
    "MOST RECENT turn -- that turn is the referent, and an earlier one becomes the referent only when " +
    "the latest question points at it explicitly. Phrasing an earlier turn happens to share with the " +
    "latest question is NOT a reference: two questions worded alike are still about their own " +
    "subjects, so a repeated 'what value should I use' asks about the most recent setting, not the " +
    "one an earlier turn asked it about. Resolve references from what the user previously ASKED; do " +
    "not import a specific parameter or setting name from the assistant's ANSWER unless the latest " +
    "question explicitly refers to it (a new topic in the latest question overrides a prior answer's " +
    "subject). Name the referent the way the USER named it -- the setting's own name, never a code " +
    "variable, formula term, or field name the assistant's answer used for it (fold in 'Batches', not " +
    "'batch_size'). Do not answer it, do not add new topics -- output only the rewritten question.";

  var STOPWORDS = new Set(("a an the of to in on for and or is are was were be been being do does did " +
    "how what which when where why who whom this that these those it its as at by with from into " +
    "can could should would will i you we they he she them his her their our your me my mine " +
    "if then than so such not no yes about over under between out up down off also just only " +
    "gem prf gemprf does use used using get set").split(/\s+/));

  var knowledgeIndex = null;
  // bm25 and chunkVectors are positionally parallel to knowledgeIndex.chunks -- all three from one corpus.
  var bm25 = null;
  var chatModels = [];
  var embedModels = [];
  // Installed-model name -> Ollama manifest digest; gates whether shipped corpus vectors are trustworthy.
  var modelDigests = {};
  var chatModel = DEFAULT_MODEL;
  var embedModel = null;
  var chunkVectors = null;
  var paramRows = null;
  var connected = false;
  var busy = false;
  // True while the corpus is being embedded; the input is held shut for the duration.
  var indexing = false;
  var history = [];
  var activeBot = null;
  var pendingClarify = null;
  var valueChoiceActive = false;
  var els = {};

  // Lowercase word tokens minus stopwords. '.' and '_' stay INSIDE a token so paper.full survives whole.
  function tokenize(s) {
    var out = [];
    var raw = (s || "").toLowerCase().match(/[a-z0-9_.]+/g) || [];
    for (var i = 0; i < raw.length; i++) {
      var t = raw[i].replace(/^[._]+|[._]+$/g, "");
      if (t.length < 2 || STOPWORDS.has(t)) continue;
      out.push(t);
    }
    return out;
  }

  function escapeHtml(s) {
    return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Minimal safe markdown: escape, then fenced/inline code, bold, paragraphs.
  function renderMarkdown(text) {
    var blocks = text.split(/```/);
    var html = "";
    for (var i = 0; i < blocks.length; i++) {
      if (i % 2 === 1) {
        var body = blocks[i].replace(/^[a-zA-Z0-9]*\n/, "");
        html += "<pre><code>" + escapeHtml(body.replace(/\n$/, "")) + "</code></pre>";
      } else {
        var seg = escapeHtml(blocks[i]);
        seg = seg.replace(/`([^`]+)`/g, "<code>$1</code>");
        seg = seg.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        var paras = seg.split(/\n{2,}/);
        for (var p = 0; p < paras.length; p++) {
          var para = paras[p].trim();
          if (para) html += "<p>" + para.replace(/\n/g, "<br>") + "</p>";
        }
      }
    }
    return html;
  }

  function prepareBm25(chunks) {
    var docs = chunks.map(function (c) {
      // The de-punctuated source_id rides along so 'paper.full' also matches a question that says 'paper'.
      var sidWords = (c.source_id || "").replace(/[._]/g, " ");
      return tokenize(c.heading + " " + sidWords + " " + c.text);
    });
    var df = Object.create(null);
    var totalLen = 0;
    for (var i = 0; i < docs.length; i++) {
      totalLen += docs[i].length;
      var seen = Object.create(null);
      for (var j = 0; j < docs[i].length; j++) {
        var t = docs[i][j];
        if (!seen[t]) { seen[t] = 1; df[t] = (df[t] || 0) + 1; }
      }
    }
    var N = docs.length;
    var tf = docs.map(function (d) {
      var m = Object.create(null);
      for (var k = 0; k < d.length; k++) m[d[k]] = (m[d[k]] || 0) + 1;
      return m;
    });
    return { tf: tf, df: df, N: N, avgdl: totalLen / (N || 1), len: docs.map(function (d) { return d.length; }) };
  }

  // One score per chunk, positionally aligned with knowledgeIndex.chunks. weightedTerms is term -> weight.
  function scoreChunksBm25(bm25Model, weightedTerms) {
    var k1 = 1.5, b = 0.75;
    var scores = new Array(bm25Model.N).fill(0);
    for (var d = 0; d < bm25Model.N; d++) {
      var s = 0;
      var dl = bm25Model.len[d];
      for (var term in weightedTerms) {
        var f = bm25Model.tf[d][term];
        if (!f) continue;
        var n = bm25Model.df[term] || 0;
        var idfv = Math.log(1 + (bm25Model.N - n + 0.5) / (n + 0.5));
        var denom = f + k1 * (1 - b + b * dl / bm25Model.avgdl);
        s += weightedTerms[term] * idfv * (f * (k1 + 1)) / denom;
      }
      scores[d] = s;
    }
    return scores;
  }

  // Absolute cosine floor for the embedding matcher only; the per-model value ships in meta.parameter_floor.
  var DEFAULT_PARAMETER_FLOOR = 0.58;
  // Which parameters the ANSWER cites: tight, or unrelated settings ride into the prompt.
  var PARAMETER_TIE_BAND = 0.02;
  var MAX_RETURNED_PARAMETERS = 4;
  // What forkCandidates may offer: wider than the band, since two live settings can score ~0.07 apart.
  var FORK_POOL_SIZE = 6;

  // Two rows per parameter (naming, meaning) so a question can match either surface; order feeds buildParamVectors.
  function buildParamRows() {
    var rows = [];
    (knowledgeIndex.parameters || []).forEach(function (p) {
      rows.push({ pid: p.id, text: p.label + ". Aliases: " + ((p.aliases || []).join(", ")) + ". Identifier: " + p.id + "." });
      rows.push({ pid: p.id, text: "XML path: " + (p.xml_path || "") + ". Summary: " + (p.summary || "") + " Significance: " + (p.significance || "") });
    });
    return rows;
  }

  // Match parameters by cosine: absolute floor, then the tie band for specs and a depth cut for the pool.
  function matchParametersEmbedding(qVec) {
    // Null-prototype map: the keys are catalog ids, and one named 'constructor' must not inherit a score.
    var bestScoreByParam = Object.create(null);
    // paramRows is two rows per parameter, so this walks surfaces (naming, meaning) rather than parameters.
    for (var i = 0; i < paramRows.length; i++) {
      // Rows and query are both L2-normalized upstream, so this dot product is already the cosine.
      var s = dot(qVec, paramRows[i].vec), pid = paramRows[i].pid;
      // Best-of, not sum: hitting either surface hard is the signal; owning two rows is not a bonus.
      if (bestScoreByParam[pid] === undefined || s > bestScoreByParam[pid]) bestScoreByParam[pid] = s;
    }
    var ordered = Object.keys(bestScoreByParam).map(function (pid) { return [pid, bestScoreByParam[pid]]; })
      .sort(function (a, b) { return b[1] - a[1]; });
    // Empty catalog or unbuilt rows: nothing to gate, and ordered[0] on the next line would throw.
    if (!ordered.length) return { specs: [], scores: {}, pool: [] };
    // The floor answers 'names a parameter at all?', the band 'which tie with the best?' -- a max() collapses them.
    if (ordered[0][1] < parameterFloor()) return { specs: [], scores: {}, pool: [] };
    var cutoff = ordered[0][1] - PARAMETER_TIE_BAND;
    var specById = Object.create(null);
    (knowledgeIndex.parameters || []).forEach(function (p) { specById[p.id] = p; });
    var specs = [], scores = {}, pool = [];
    // Pool depth ends the walk: past FORK_POOL_SIZE ranks, nothing can enter either list.
    for (var k = 0; k < ordered.length && pool.length < FORK_POOL_SIZE; k++) {
      var spec = specById[ordered[k][0]];
      // A scored id with no spec means paramRows outlived the catalog -- skip it rather than emit a hole.
      if (!spec) continue;
      // Rank alone earns a pool slot: a fork asks the user, so a near-miss is still worth offering.
      pool.push(spec);
      // Citing is stricter -- inside the tie band, and capped so one question cannot drag in a whole subsystem.
      if (ordered[k][1] >= cutoff && specs.length < MAX_RETURNED_PARAMETERS) {
        specs.push(spec); scores[spec.id] = ordered[k][1];
      }
    }
    return { specs: specs, scores: scores, pool: pool };
  }

  // Ordered words, duplicates kept -- unlike disambigTokens, which dedupes and drops anything under 3 chars.
  function nameWords(text) {
    return String(text || "").toLowerCase().match(/[a-z0-9_]+/g) || [];
  }

  // Name matching, the net under the embedder: `named` is a whole name, `partial` a head-anchored run.
  function matchParametersByName(question) {
    // Single-space joins on both sides, so every indexOf below is word-boundary anchored by construction.
    var qwords = nameWords(question), qjoin = " " + qwords.join(" ") + " ";
    var named = [], partial = [];                      // {spec, score} per parameter, by strength of the hit
    (knowledgeIndex.parameters || []).forEach(function (p) {
      var probes = [p.label].concat(p.aliases || []);
      var hits = 0, longestProbe = 0, longestRun = 0;  // whole names that appeared, and the best partial run
      for (var i = 0; i < probes.length; i++) {
        var pw = nameWords(probes[i]);
        if (!pw.length) continue;
        // The whole name, verbatim. Single-word probes keep the >= 3 floor so an "R" cannot fire on anything.
        if (pw.length === 1 ? (pw[0].length >= 3 && qwords.indexOf(pw[0]) !== -1)
                            : qjoin.indexOf(" " + pw.join(" ") + " ") !== -1) {
          hits++; longestProbe = Math.max(longestProbe, probes[i].length); continue;
        }
        // Else the longest suffix run of >= 2 words: leading qualifiers alone name nothing.
        for (var start = 0; start <= pw.length - 2; start++) {
          if (qjoin.indexOf(" " + pw.slice(start).join(" ") + " ") !== -1) { longestRun = Math.max(longestRun, pw.length - start); break; }
        }
      }
      // hits dominates (x1000) so probe count outranks length; longestProbe only separates equal-hit parameters.
      if (hits > 0) named.push({ spec: p, score: hits * 1000 + longestProbe });
      // A whole name always beats a partial one, so a parameter that has one is never listed as the other.
      else if (longestRun > 0) partial.push({ spec: p, score: longestRun });
    });
    var rank = function (list) {
      return list.sort(function (a, b) { return b.score - a.score; })
        .slice(0, MAX_RETURNED_PARAMETERS).map(function (m) { return m.spec; });
    };
    return { named: rank(named), partial: rank(partial) };
  }

  var _tokenOwnersCache = null;
  var _anchorVocabCache = null;
  var _headOwnersCache = null;

  // Content tokens (>2 chars) for term disambiguation; unlike tokenize() it splits on '.' and keeps stopwords.
  function disambigTokens(text) {
    var out = Object.create(null);
    (String(text || "").toLowerCase().match(/[a-z0-9_]+/g) || []).forEach(function (t) {
      if (t.length > 2) out[t] = 1;
    });
    return Object.keys(out);
  }

  // Map each catalog term to the parameter ids that use it.
  function tokenOwners() {
    if (_tokenOwnersCache) return _tokenOwnersCache;
    var owners = Object.create(null);
    (knowledgeIndex.parameters || []).forEach(function (p) {
      disambigTokens([p.label].concat(p.aliases || []).join(" ")).forEach(function (t) {
        (owners[t] = owners[t] || Object.create(null))[p.id] = 1;
      });
    });
    _tokenOwnersCache = owners;
    return owners;
  }

  // Head nouns (last word of a label/alias) keyed to the parameters they HEAD, not the ones they merely contain.
  function headOwners() {
    if (_headOwnersCache) return _headOwnersCache;
    var heads = Object.create(null);
    (knowledgeIndex.parameters || []).forEach(function (p) {
      [p.label].concat(p.aliases || []).forEach(function (entry) {
        var words = String(entry || "").toLowerCase().match(/[a-z0-9_]+/g) || [];
        var last = words.length ? words[words.length - 1] : null;
        if (last && last.length > 2) (heads[last] = heads[last] || Object.create(null))[p.id] = 1;
      });
    });
    _headOwnersCache = heads;
    return heads;
  }

  // Fork the parameters an ambiguous term leaves open. poolSpecs is the fork pool, not the answer's specs.
  function forkCandidates(question, poolSpecs) {
    var owners = tokenOwners();
    var heads = headOwners();
    var ownerSets = disambigTokens(question).filter(function (t) { return owners[t]; }).map(function (t) { return owners[t]; });
    // Only a head noun may CREATE a fork; a family qualifier ('stimulus') may only narrow one.
    var seedSets = disambigTokens(question).filter(function (t) {
      return heads[t] && Object.keys(heads[t]).length >= 2;
    }).map(function (t) { return heads[t]; });
    if (!seedSets.length) return [];
    var candSet = Object.create(null);
    Object.keys(seedSets[0]).forEach(function (pid) {
      if (seedSets.every(function (o) { return o[pid]; })) candSet[pid] = 1;
    });
    if (Object.keys(candSet).length < 2) return [];
    ownerSets.forEach(function (o) {
      var keys = Object.keys(candSet);
      var overlap = keys.filter(function (pid) { return o[pid]; });
      // Narrow only on a proper, non-empty overlap: a term covering all candidates (or none) discriminates nothing.
      if (overlap.length > 0 && overlap.length < keys.length) {
        candSet = Object.create(null);
        overlap.forEach(function (pid) { candSet[pid] = 1; });
      }
    });
    // A fork may only offer parameters retrieval judged plausible -- never one the matcher did not surface.
    var forkSpecs = (poolSpecs || []).filter(function (spec) { return candSet[spec.id]; });
    if (forkSpecs.length < 2) return [];
    // Candidates under one parent are facets of one setting; only a term spanning two parents forks.
    var parents = Object.create(null);
    forkSpecs.forEach(function (spec) { parents[String(spec.id).split(".").slice(0, -1).join(".")] = 1; });
    if (Object.keys(parents).length < 2) return [];
    return forkSpecs.slice(0, 4);
  }

  var FORK_SUBJECT_SYSTEM =
    "You triage questions for a GEM-pRF configuration assistant. A word in the user's message matches " +
    "more than one configuration setting, and the assistant must decide whether to ask which one they " +
    "mean or simply answer.\n" +
    "Answer YES only if one of those settings is what the user is ASKING ABOUT -- so that knowing which " +
    "one they mean would change the answer.\n" +
    "Answer NO when the word appears only in passing and the question is really about something else: " +
    "what a tool, script, page or file does; how something behaves, runs or is installed; a licence, a " +
    "citation, a version. In those cases the question has its own subject and the matching word is " +
    "merely part of the description.\n" +
    "Reply with exactly one word: YES or NO.";

  // True when the forked settings are what the question is ABOUT. Fails closed; `complete` is injectable for tests.
  function forkIsTheSubject(question, fork, complete) {
    if (!fork || !fork.length) return Promise.resolve(true);
    var labels = fork.map(function (spec) { return spec.label; }).join(", ");
    return (complete || chatComplete)([
      { role: "system", content: FORK_SUBJECT_SYSTEM },
      { role: "user", content: "Settings the word matches: " + labels + "\n\nQuestion: " + question }
    ]).then(function (t) { return (t || "").trim().toUpperCase().indexOf("NO") !== 0; })
      .catch(function () { return true; });
  }

  // Every word a parameter answers to, across its label and aliases.
  function surfaceWords(spec) {
    var out = Object.create(null);
    [spec.label].concat(spec.aliases || []).forEach(function (s) {
      nameWords(s).forEach(function (w) { out[w] = 1; });
    });
    return out;
  }

  // Levenshtein distance, abandoned once the whole row exceeds `max`.
  function withinEdits(word, target, max) {
    if (Math.abs(word.length - target.length) > max) return false;
    var prev = [], cur = [], j;
    for (j = 0; j <= target.length; j++) prev[j] = j;
    for (var i = 1; i <= word.length; i++) {
      cur[0] = i;
      var rowBest = i;
      for (j = 1; j <= target.length; j++) {
        var cost = word.charAt(i - 1) === target.charAt(j - 1) ? 0 : 1;
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
        if (cur[j] < rowBest) rowBest = cur[j];
      }
      // Whole row over budget, so no continuation comes back under it.
      if (rowBest > max) return false;
      var swap = prev; prev = cur; cur = swap;
    }
    return prev[target.length] <= max;
  }

  // No edits under 5 chars, where one edit reaches a different word ('grid'/'grip').
  function editBudget(word) { return word.length >= 8 ? 2 : word.length >= 5 ? 1 : 0; }

  // Index of the reply word matching a surface word outright or by a spelling slip ('spetial' is 'spatial'); -1 when none.
  function replySaysAt(replyWords, surfaceWord) {
    var budget = editBudget(surfaceWord);
    for (var i = 0; i < replyWords.length; i++) {
      if (replyWords[i] === surfaceWord) return i;
      if (budget && withinEdits(replyWords[i], surfaceWord, budget)) return i;
    }
    return -1;
  }

  function replySays(replyWords, surfaceWord) { return replySaysAt(replyWords, surfaceWord) !== -1; }

  // True when 'not' sits within two words before words[idx] ('not the stimulus one'); 'not sure, maybe stimulus' is too far.
  function negatedAt(words, idx) {
    for (var i = Math.max(0, idx - 2); i < idx; i++) { if (words[i] === "not") return true; }
    return false;
  }

  var ORDINAL_WORDS = { first: 0, "1st": 0, second: 1, "2nd": 1, third: 2, "3rd": 2 };

  // Menu-answer filler. Deliberately not in STOPWORDS, which would change every BM25 score.
  var SELECTION_NOISE = new Set("one ones option options setting settings choice choose pick mean meant want please".split(" "));

  // People answer a two-item menu positionally. The menu order IS fork order, which is what they saw.
  function resolveForkOrdinal(reply, fork) {
    var words = nameWords(reply), idx = null;
    for (var i = 0; i < words.length && idx === null; i++) {
      if (ORDINAL_WORDS[words[i]] !== undefined) {
        // 'not the first one' rejects fork[0]: a pair inverts to the other, a longer menu falls through to re-ask.
        if (negatedAt(words, i)) return (fork.length === 2 && ORDINAL_WORDS[words[i]] < 2) ? fork[1 - ORDINAL_WORDS[words[i]]] : null;
        idx = ORDINAL_WORDS[words[i]];
      }
    }
    // A bare number only: '2' alone is a choice, but the 2 in 'about 2 degrees' is a value.
    if (idx === null && /^\s*[1-9]\s*$/.test(reply)) idx = parseInt(reply, 10) - 1;
    return (idx !== null && idx >= 0 && idx < fork.length) ? fork[idx] : null;
  }

  // Which candidate a reply picks, by the words that tell them apart; null when it names none or splits.
  function resolveForkReply(reply, fork) {
    if (!fork || fork.length < 2) return null;
    // Positional answers carry no label words at all, so they must be read before the word scoring.
    var byOrdinal = resolveForkOrdinal(reply, fork);
    if (byOrdinal) return byOrdinal;
    var replyWords = nameWords(reply);
    var surfaces = fork.map(surfaceWords);
    // A word every candidate answers to cannot discriminate -- 'radius' is both a grid label and a stimulus alias.
    var discriminates = function (w) { return !surfaces.every(function (s) { return s[w]; }); };
    var best = null, bestScore = 0, tied = false, rejected = [];
    surfaces.forEach(function (surf, i) {
      var named = 0, negated = 0;
      Object.keys(surf).forEach(function (w) {
        if (!discriminates(w)) return;
        var at = replySaysAt(replyWords, w);
        if (at < 0) return;
        if (negatedAt(replyWords, at)) negated++; else named++;
      });
      // A candidate only ever mentioned under negation is a rejection, not a pick.
      if (negated > 0 && named === 0) rejected.push(i);
      if (named > bestScore) { bestScore = named; best = fork[i]; tied = false; }
      else if (named === bestScore && named > 0) tied = true;
    });
    if (best && !tied) return best;
    // 'not the stimulus one' on a pair picks the other; a longer menu re-asks instead of guessing.
    if (bestScore === 0 && rejected.length === 1 && fork.length === 2) return fork[1 - rejected[0]];
    return null;
  }

  // Fold the reply in as a correction naming the pick, not as an appended annotation.
  function foldClarifyReply(original, reply, fork, asked) {
    var picked = resolveForkReply(reply, fork);
    // Restate the question with the reply, since '8' is inert alone.
    if (!picked) {
      return (asked && !(fork && fork.length >= 2))
        ? original + " -- in answer to \"" + asked + "\": " + reply
        : original + " (" + reply + ")";
    }
    // Rejected candidates are referred to, never named, or each becomes a named hit next turn.
    var others = fork.length - 1;
    var folded = original + " -- I mean the " + picked.label + " setting, not " +
      (others === 1 ? "the other one" : "the others") + ".";
    // Keep the reply too unless it did nothing but select -- 'stimulus, 10 degrees' still carries the 10 degrees.
    var extra = nameWords(reply).filter(function (w) { return !STOPWORDS.has(w); });
    var surfWords = Object.keys(surfaceWords(picked));
    // Same tolerance the resolver used, or a reply it read as a pick is echoed back as leftover content.
    var selects = function (w) {
      if (SELECTION_NOISE.has(w) || ORDINAL_WORDS[w] !== undefined || /^[1-9]$/.test(w)) return true;
      return surfWords.some(function (s) { return w === s || (editBudget(s) && withinEdits(w, s, editBudget(s))); });
    };
    var onlySelects = extra.length > 0 && extra.every(selects);
    return folded + (onlySelects ? "" : " " + reply);
  }

  // Reply words every candidate answers to: matched, but unable to pick ('visual' names both radii).
  function forkSharedWords(reply, fork) {
    var surfaces = fork.map(surfaceWords);
    return nameWords(reply).filter(function (w) {
      if (STOPWORDS.has(w) || SELECTION_NOISE.has(w)) return false;
      return surfaces.every(function (surf) {
        return Object.keys(surf).some(function (s) { return w === s || (editBudget(s) && withinEdits(w, s, editBudget(s))); });
      });
    });
  }

  // One label word per candidate no other candidate answers to; catalog-rarest, so it names little else.
  function forkHintWords(fork) {
    var owners = tokenOwners();
    var rarity = function (w) { return Object.keys(owners[w] || {}).length; };
    return fork.map(function (spec, i) {
      var others = fork.filter(function (_, j) { return j !== i; }).map(surfaceWords);
      var unique = nameWords(spec.label).filter(function (w) {
        return w.length > 2 && !STOPWORDS.has(w) && !others.some(function (surf) { return surf[w]; });
      });
      unique.sort(function (a, b) { return rarity(a) - rarity(b); });
      return unique[0] || null;
    });
  }

  // Second ask, numbered so a positional answer works; a reply that named ALL candidates is taught a winnable word.
  function forkReaskQuestion(fork, reply) {
    var numbered = fork.map(function (s, i) { return (i + 1) + ") " + s.label; }).join("   ");
    var shared = reply ? forkSharedWords(reply, fork) : [];
    if (!shared.length) return "Sorry -- I couldn't tell which of those you meant. Pick one by name or number: " + numbered;
    var hints = forkHintWords(fork);
    var say = hints.every(Boolean) ? "say \"" + hints.join("\" or \"") + "\", or " : "";
    return "\"" + shared.join(" ") + "\" matches " + (fork.length === 2 ? "both" : "all") + " of those, so it can't pick one -- " +
      say + "pick a number: " + numbered;
  }

  function forkQuestion(fork) {
    var labels = fork.map(function (spec) { return spec.label; });
    var joined = labels.length === 2 ? labels[0] + " or " + labels[1]
      : labels.slice(0, -1).join(", ") + ", or " + labels[labels.length - 1];
    return "GEM-pRF has more than one setting that matches that -- which do you mean: " + joined + "?";
  }

  // Three surfaces from labels+aliases: single-word tokens, whole phrases, and a glue form ('maxsigma').
  function anchorVocab() {
    if (_anchorVocabCache) return _anchorVocabCache;
    var tokens = Object.create(null), phrases = [], collapsed = [];
    (knowledgeIndex.parameters || []).forEach(function (p) {
      [p.label].concat(p.aliases || []).forEach(function (entry) {
        var words = disambigTokens(entry);
        if (words.length === 1) {
          tokens[words[0]] = 1;
          if (words[0].indexOf("_") !== -1) collapsed.push(words[0].replace(/_/g, ""));
        } else if (words.length > 1) {
          phrases.push(entry.toLowerCase());
          collapsed.push(words.join(""));
        }
      });
    });
    _anchorVocabCache = { tokens: tokens, phrases: phrases, collapsed: collapsed.filter(function (c) { return c.length >= 7; }) };
    return _anchorVocabCache;
  }

  // True when the question anchors to one specific setting.
  function namesParameter(question) {
    var vocab = anchorVocab();
    var low = String(question || "").toLowerCase();
    if (disambigTokens(question).some(function (t) { return vocab.tokens[t]; })) return true;
    if (vocab.phrases.some(function (ph) { return low.indexOf(ph) !== -1; })) return true;
    var compact = low.replace(/[^a-z0-9]/g, "");
    return vocab.collapsed.some(function (c) { return compact.indexOf(c) !== -1; });
  }

  // Fixed value-choice clarify asking the user's goal (fallback template).
  function valueClarifyFallback(fork, namesSetting) {
    var goal = "what are you optimizing for -- " + GOAL_TRIAD + "?";
    // One source for the fork sentence, shared with generateValueClarify, so the two wordings cannot drift.
    if (fork.length) return forkQuestion(fork) + " And " + goal;
    if (namesSetting) return "That depends on your goal -- " + goal + " (Or say 'just the default' for the documented value.)";
    return "Which setting are you choosing a value for, and " + goal + " (Or name the setting and say 'just the default'.)";
  }

  // Drop an ordinary leading capital so a sentence reads after "... And"; an acronym ("GPU memory") keeps its.
  function uncapitalize(text) {
    return /^[A-Z][a-z]/.test(text) ? text.charAt(0).toLowerCase() + text.slice(1) : text;
  }

  function generateValueClarify(question, retrieval, fork) {
    var focusSpecs = fork.length ? fork : (retrieval.specs || []).slice(0, 1);
    // Nothing matched: tailoring to a "- none" setting block invents an angle, so ask the fixed question instead.
    if (!focusSpecs.length) return Promise.resolve(valueClarifyFallback(fork, namesParameter(question)));
    // Wide enough to tell a rule from a bare direction; 260 chars truncated the deciding sentence.
    var evidence = (retrieval.chunks || []).slice(0, 5)
      .map(function (c) { return "- " + (c.heading || c.source_id) + ": " + c.text.split(/\s+/).join(" ").slice(0, 600); })
      .join("\n") || "- none";
    var human = "User question: " + question + "\n\nThe setting the user is asking about:\n" +
      parameterContext(focusSpecs) + "\n\nWhat the documentation says:\n" + evidence +
      // Which setting they mean is prepended below, so the model is told not to spend its one question on it.
      (fork.length ? "\n\nWhich of these the user means is ALREADY being asked, ahead of your question -- do " +
        "not ask it again: " + fork.map(function (s) { return s.label; }).join(", ") : "") +
      "\n\nYour one clarifying question:";
    return chatComplete([
      { role: "system", content: VALUE_CLARIFY_SYSTEM },
      { role: "user", content: human }
    ]).then(function (t) {
      t = (t || "").replace(/^["'`]+|["'`]+$/g, "").trim();
      if (!t) return valueClarifyFallback(fork, namesParameter(question));
      // Only ever ONE clarify turn, so the fork sentence is prepended rather than left to the model.
      return fork.length ? forkQuestion(fork) + " And " + uncapitalize(t) : t;
    }).catch(function () { return valueClarifyFallback(fork, namesParameter(question)); });
  }

  function embedPrefix(model, kind) {
    var m = (model || "").toLowerCase();
    var key = Object.keys(EMBED_PREFIX).filter(function (k) { return m.indexOf(k) !== -1; })[0];
    return key ? EMBED_PREFIX[key][kind] : "";
  }

  function l2normalize(arr) {
    var v = new Float32Array(arr.length);
    var norm = 0;
    for (var i = 0; i < arr.length; i++) { v[i] = arr[i]; norm += arr[i] * arr[i]; }
    norm = Math.sqrt(norm) || 1;
    for (var j = 0; j < v.length; j++) v[j] /= norm;
    return v;
  }

  // Every vector reaching here is L2-normalized, so this dot product IS the cosine.
  function dot(a, b) {
    var s = 0, n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }

  // int8: full-float JSON overflows the ~5MB localStorage quota, and ~1/127 error is negligible here.
  function encodeVectors(vecs) {
    if (!vecs.length) return "0:";
    var dim = vecs[0].length, i8 = new Int8Array(vecs.length * dim);
    for (var i = 0; i < vecs.length; i++) {
      var v = vecs[i];
      for (var j = 0; j < dim; j++) {
        var q = Math.round(v[j] * 127);
        i8[i * dim + j] = q > 127 ? 127 : (q < -128 ? -128 : q);
      }
    }
    // 32k chars at a time: String.fromCharCode.apply blows the argument limit on a full corpus in one call.
    var u8 = new Uint8Array(i8.buffer), CH = 0x8000, s = "";
    for (var k = 0; k < u8.length; k += CH) s += String.fromCharCode.apply(null, u8.subarray(k, k + CH));
    return dim + ":" + btoa(s);
  }

  function decodeVectors(str, expectedCount) {
    try {
      var sep = str.indexOf(":");
      if (sep < 0) return null;
      var dim = parseInt(str.slice(0, sep), 10);
      if (!dim) return expectedCount === 0 ? [] : null;
      var bin = atob(str.slice(sep + 1)), u8 = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      var i8 = new Int8Array(u8.buffer), count = i8.length / dim;
      if (count !== Math.floor(count) || (expectedCount != null && count !== expectedCount)) return null;
      var out = new Array(count);
      for (var c = 0; c < count; c++) {
        var v = new Float32Array(dim);
        for (var d = 0; d < dim; d++) v[d] = i8[c * dim + d] / 127;
        out[c] = v;
      }
      return out;
    } catch (e) { return null; }
  }

  // Embed texts via the visitor's Ollama (batch, legacy fallback). Rows, chunks and queries all fold here.
  function ollamaEmbed(model, texts, kind) {
    var pref = embedPrefix(model, kind);
    // Case-folded: nomic cosines a Title Case string at 0.55 against its own lowercase form.
    var inputs = texts.map(function (t) { return pref + String(t).toLowerCase(); });
    return fetch(OLLAMA + "/api/embed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model, input: inputs })
    }).then(function (r) {
      if (!r.ok) throw new Error("embed HTTP " + r.status);
      return r.json();
    }).then(function (j) {
      if (j && j.embeddings && j.embeddings.length === inputs.length) return j.embeddings.map(l2normalize);
      throw new Error("unexpected /api/embed response");
    }).catch(function () {
      return Promise.all(inputs.map(function (t) {
        return fetch(OLLAMA + "/api/embeddings", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: model, prompt: t })
        }).then(function (r) {
          if (!r.ok) throw new Error("embeddings HTTP " + r.status);
          return r.json();
        }).then(function (j) { return l2normalize(j.embedding); });
      }));
    });
  }

  function pickEmbedModel(available) {
    if (!available.length) return null;
    if (available.indexOf(PREFERRED_EMBED) !== -1) return PREFERRED_EMBED;
    var order = ["nomic-embed-text", "mxbai-embed-large", "bge", "arctic", "e5", "minilm"];
    for (var i = 0; i < order.length; i++) {
      for (var j = 0; j < available.length; j++) {
        if (available[j].toLowerCase().indexOf(order[i]) !== -1) return available[j];
      }
    }
    return available[0];
  }

  // Old-corpus/old-model blobs are never read again; left behind they eat the quota until setItem fails silently.
  function purgeStaleVectors(prefix, keep) {
    var stale = [];
    try {
      for (var i = 0; i < localStorage.length; i++) {
        var k = localStorage.key(i);
        if (k && k.indexOf(prefix) === 0 && k !== keep) stale.push(k);
      }
      stale.forEach(function (k) { localStorage.removeItem(k); });
    } catch (e) {  }
  }

  var vectorsFilePromise = null;
  function fetchVectorsFile() {
    if (!vectorsFilePromise) {
      vectorsFilePromise = fetch(VECTORS_URL, { cache: "no-cache" })
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { return null; });
    }
    return vectorsFilePromise;
  }

  // Query vectors come from the visitor's model, so shipped doc vectors need that exact model: same digest, same corpus.
  function usablePrecomputed(p) {
    return p && p.corpus_sha === knowledgeIndex.meta.corpus_sha &&
      p.model && embedModel && embedModel.toLowerCase().indexOf(p.model) !== -1 &&
      p.digest && modelDigests[embedModel] === p.digest ? p : null;
  }

  function saveVectors(cacheKey, vecs) {
    try { localStorage.setItem(cacheKey, encodeVectors(vecs)); } catch (e) {  }
  }

  // Embed the corpus chunks once, cached by (corpus_sha, model, digest): re-pulling a model under the same name must miss.
  function buildChunkVectors(onCacheMiss) {
    var cacheKey = "gpa:vec4:" + knowledgeIndex.meta.corpus_sha + ":" + embedModel + ":" + (modelDigests[embedModel] || "");
    purgeStaleVectors("gpa:vec", cacheKey);
    try {
      var decoded = decodeVectors(localStorage.getItem(cacheKey) || "", knowledgeIndex.chunks.length);
      if (decoded) { chunkVectors = decoded; return Promise.resolve(true); }
    } catch (e) {  }

    if (onCacheMiss) onCacheMiss();
    return fetchVectorsFile().then(function (p) {
      var shipped = usablePrecomputed(p) && decodeVectors(p.chunks || "", knowledgeIndex.chunks.length);
      if (shipped) return shipped;
      var texts = knowledgeIndex.chunks.map(function (c) { return c.heading + "\n" + c.text; });
      return ollamaEmbed(embedModel, texts, "doc");
    }).then(function (vecs) {
      chunkVectors = vecs;
      saveVectors(cacheKey, vecs);
      return true;
    });
  }

  // Cached by POSITION and guarded only by row count, so buildParamRows must stay deterministic.
  function buildParamVectors() {
    var rows = buildParamRows();
    if (!rows.length) { paramRows = []; return Promise.resolve(true); }
    var cacheKey = "gpa:pvec4:" + knowledgeIndex.meta.corpus_sha + ":" + embedModel + ":" + (modelDigests[embedModel] || "");
    purgeStaleVectors("gpa:pvec", cacheKey);
    try {
      var decoded = decodeVectors(localStorage.getItem(cacheKey) || "", rows.length);
      if (decoded) { for (var i = 0; i < rows.length; i++) rows[i].vec = decoded[i]; paramRows = rows; return Promise.resolve(true); }
    } catch (e) {  }
    return fetchVectorsFile().then(function (p) {
      var shipped = usablePrecomputed(p) && decodeVectors(p.params || "", rows.length);
      if (shipped) return shipped;
      return ollamaEmbed(embedModel, rows.map(function (r) { return r.text; }), "doc");
    }).then(function (vecs) {
      for (var j = 0; j < rows.length; j++) rows[j].vec = vecs[j];
      paramRows = rows;
      saveVectors(cacheKey, vecs);
      return true;
    }).catch(function () { paramRows = null; return false; });
  }

  function minmaxNormalize(arr) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < arr.length; i++) { if (arr[i] < lo) lo = arr[i]; if (arr[i] > hi) hi = arr[i]; }
    var rng = hi - lo, out = new Array(arr.length);
    for (var j = 0; j < arr.length; j++) out[j] = rng > 0 ? (arr[j] - lo) / rng : 0;
    return out;
  }

  // Per-model cosine floor from the index: each model sits on its own band (nomic off-topic ~0.52, not ~0).
  function modelFloorEntry(tableName) {
    var table = (knowledgeIndex && knowledgeIndex.meta && knowledgeIndex.meta[tableName]) || null;
    if (!table) return null;
    var m = (embedModel || "").toLowerCase();
    var key = Object.keys(table).filter(function (k) { return k !== "default" && m.indexOf(k) !== -1; })[0];
    if (key && typeof table[key] === "number") return { value: table[key], calibrated: true };
    return typeof table["default"] === "number" ? { value: table["default"], calibrated: false } : null;
  }

  function modelFloor(tableName, fallback) {
    var entry = modelFloorEntry(tableName);
    return entry ? entry.value : fallback;
  }

  // Below this the corpus holds no evidence; the fallback sits low so an unmeasured model under-refuses.
  var DEFAULT_EVIDENCE_FLOOR = 0.55;
  function evidenceFloor() { return modelFloor("evidence_floor", DEFAULT_EVIDENCE_FLOOR); }

  // True only when meta ships a floor measured for THIS model; the 'default' entry is not a measurement.
  function evidenceFloorCalibrated() {
    var entry = modelFloorEntry("evidence_floor");
    return !!(entry && entry.calibrated);
  }

  // Below this, the question is judged to name no parameter -- see matchParametersEmbedding.
  function parameterFloor() { return modelFloor("parameter_floor", DEFAULT_PARAMETER_FLOOR); }

  // Min-max normalises each side, so absolute similarity does not survive this call -- read the raw arrays first.
  function blendScores(bm25Scores, semScores) {
    if (!semScores) return bm25Scores;
    var nb = minmaxNormalize(bm25Scores), ns = minmaxNormalize(semScores);
    var out = new Array(bm25Scores.length);
    for (var i = 0; i < out.length; i++) out[i] = HYBRID_ALPHA * ns[i] + (1 - HYBRID_ALPHA) * nb[i];
    return out;
  }

  // Borrowed-vocabulary weight for query expansion; the user's own words carry 1.
  var EXPANSION_WEIGHT = 0.6;
  // A literal name outranks every cosine the embedder can produce.
  var NAME_MATCH_SCORE = 1.0;
  // Per-source cap: at TOP_K 6, a cap of 3 let one source hold half the evidence. See tests/RETRIEVAL.md.
  var MAX_PER_SOURCE = 2;

  // How many top cosines the abstain gate averages; the single-term exemption lives in selectChunks.
  var ABSTAIN_SAMPLE = 5;

  // Mean of the n best scores. Reads the neighbourhood around the top hit, not the hit alone.
  function meanTopN(scores, n) {
    var sorted = scores.slice().sort(function (a, b) { return b - a; });
    var take = Math.min(n, sorted.length);
    if (!take) return 0;
    var sum = 0;
    for (var i = 0; i < take; i++) sum += sorted[i];
    return sum / take;
  }

  // Embed the question, or say why not: 'unconfigured' (nothing to score against) vs 'embed-failed'.
  function embedQuery(question) {
    if (!embedModel || (!chunkVectors && !paramRows)) return Promise.resolve({ vec: null, degraded: "unconfigured" });
    // "query" selects the search_query prefix, not the search_document one used at index time.
    return ollamaEmbed(embedModel, [question], "query")
      // The catch collapses a failed embed into a degraded turn rather than a dead one.
      .then(function (qv) { return { vec: qv[0], degraded: null }; })
      .catch(function () { return { vec: null, degraded: "embed-failed" }; });
  }

  // Fuse the matchers into one ranking: names always run as the net under cosine, cosine when it can.
  function matchParameters(question, qVec) {
    var byName = matchParametersByName(question);
    var cosine = (qVec && paramRows) ? matchParametersEmbedding(qVec) : { specs: [], scores: {}, pool: [] };
    // Null-prototype so a parameter id like 'constructor' cannot collide.
    var scores = Object.create(null);
    var fused = [];
    cosine.specs.forEach(function (s) { scores[s.id] = cosine.scores[s.id]; fused.push(s); });
    // A literal name is confident: nomic embeds rare acronyms (nDCT) poorly, so this overwrites a real cosine.
    byName.named.forEach(function (s) {
      if (scores[s.id] === undefined) fused.push(s);
      scores[s.id] = NAME_MATCH_SCORE;
    });
    // Sort is stable, so equal scores keep insertion order: cosine rank, then name rank.
    fused.sort(function (a, b) { return scores[b.id] - scores[a.id]; });
    // Pool = cosine order, then rescues and partials; a partial may ask which setting is meant, never cite.
    var forkPool = cosine.pool.slice();
    var inPool = Object.create(null);
    forkPool.forEach(function (s) { inPool[s.id] = 1; });
    fused.concat(byName.partial).forEach(function (s) { if (!inPool[s.id]) { inPool[s.id] = 1; forkPool.push(s); } });
    // The citing cut comes AFTER the pool merge: forkPool keeps the candidates it drops.
    return { cited: fused.slice(0, MAX_RETURNED_PARAMETERS), forkPool: forkPool, scores: scores };
  }

  // term -> weight maps for the two BM25 passes; null-prototype so a token like 'constructor' cannot collide.
  function expansionWeights(question, specs) {
    var own = Object.create(null);
    tokenize(question).forEach(function (t) { own[t] = 1; });
    var expanded = Object.create(null);
    Object.keys(own).forEach(function (t) { expanded[t] = own[t]; });
    // Borrowed terms actually added, for callers that want to see why a chunk ranked.
    var words = [];
    // Borrow vocabulary from matched parameters so their chunks score even if worded differently.
    specs.forEach(function (p) {
      tokenize([p.label, (p.aliases || []).join(" "), p.summary].join(" ")).forEach(function (t) {
          if (!(t in expanded)) words.push(t);
        // max() rather than assignment: expansion may never dilute a term the user typed.
        expanded[t] = Math.max(expanded[t] || 0, EXPANSION_WEIGHT);
      });
    });
    return { own: own, expanded: expanded, words: words };
  }

  // Own words, plus at most that much again borrowed: expansion may amplify a chunk, never carry one in alone.
  function cappedExpansionScores(w) {
    var ownScores = scoreChunksBm25(bm25, w.own);
    var expandedScores = scoreChunksBm25(bm25, w.expanded);
    return ownScores.map(function (own, i) { return own + Math.min(expandedScores[i] - own, own); });
  }

  // Per-chunk scores; semScores null means lexical-only, which also selects the 'bm25' mode label.
  function scoreEvidence(question, qVec, citedSpecs) {
    var w = expansionWeights(question, citedSpecs);
    return {
      bm25Scores: cappedExpansionScores(w),
      semScores: (qVec && chunkVectors) ? chunkVectors.map(function (cv) { return dot(qVec, cv); }) : null,
      expansion: w.words,
      // Own terms only: expansion would inflate every query past the gate's single-term exemption.
      ownTerms: Object.keys(w.own).length,
      // Lexical anchor to the catalog ('nDCT value'): the abstain gate lets BM25 net these, not the mean.
      namesParam: namesParameter(question)
    };
  }

  // Trim a score-ranked list to `limit` with at most `cap` chunks per source -- the Python sibling is _diverse_topk.
  function diverseTopK(ranked, limit, cap) {
    var perSource = Object.create(null);
    var picked = [];
    for (var r = 0; r < ranked.length && picked.length < limit; r++) {
      var c = knowledgeIndex.chunks[ranked[r].chunkIndex];
      var sid = c.source_id || "?";
      if ((perSource[sid] || 0) >= cap) continue;
      perSource[sid] = (perSource[sid] || 0) + 1;
      picked.push(c);
    }
    return picked;
  }

  // Evidence floor, blend, rank, diversity cut. empty:true means the corpus holds nothing for this question.
  function selectChunks(evidence) {
    // Maxima must be read before blendScores: min-max maps the best cosine to 1.0, erasing magnitude.
    var maxCos = evidence.semScores ? Math.max.apply(null, evidence.semScores) : 0;
    var maxBm25 = evidence.bm25Scores.length ? Math.max.apply(null, evidence.bm25Scores) : 0;
    // A stray chunk clears the floor off-topic (sky peaks at 0.62); real coverage lifts its neighbours too.
    var topCos = evidence.semScores ? meanTopN(evidence.semScores, ABSTAIN_SAMPLE) : 0;
    // One-term and catalog-name lookups ('nDCT value') hit hard but drag no neighbours; BM25 nets them instead.
    var noNeighbourhood = topCos < evidenceFloor() && evidence.ownTerms !== 1 && !evidence.namesParam;
    // Alien vocabulary refuses only when typed terms exist AND the embedder lands nowhere -- 'binarisation' cosines in.
    var noLexicalMatch = maxBm25 <= 0 && evidence.ownTerms > 0 && maxCos < evidenceFloor();
    // An unmeasured model's band is unknown: it keeps the conjunctive max form, under-refusing as the fallback intends.
    var gated = evidenceFloorCalibrated()
      ? (noNeighbourhood || noLexicalMatch)
      : (maxCos < evidenceFloor() && maxBm25 <= 0);
    // The gate needs cosine, so it still never fires in bm25 mode.
    if (evidence.semScores && gated) {
      return { chunks: [], maxCos: maxCos, maxBm25: maxBm25, topCos: topCos, empty: true };
    }
    var blended = blendScores(evidence.bm25Scores, evidence.semScores);
    var ranked = [];
    for (var i = 0; i < blended.length; i++) {
      if (blended[i] > 0) ranked.push({ chunkIndex: i, score: blended[i] });
    }
    ranked.sort(function (a, b) { return b.score - a.score; });
    return { chunks: diverseTopK(ranked, TOP_K, MAX_PER_SOURCE), maxCos: maxCos, maxBm25: maxBm25, topCos: topCos, empty: false };
  }

  // The only place the record is built, so the gated-empty shape cannot drift from the full one.
  function retrievalResult(q, params, evidence, picked) {
    // The floor gate empties the parameters too: an off-topic question must not seed a fork clarify.
    var gated = picked.empty;
    return {
      chunks: picked.chunks,
      specs: gated ? [] : params.cited,
      forkPool: gated ? [] : params.forkPool,
      paramScores: gated ? {} : params.scores,
      expansion: gated ? [] : evidence.expansion,
      mode: evidence.semScores ? "hybrid" : "bm25",
      degraded: q.degraded,
      maxCos: picked.maxCos, maxBm25: picked.maxBm25, topCos: picked.topCos
    };
  }

  // Retrieve evidence + matched params: hybrid, or BM25-only without embeddings.
  function retrieve(question) {
    return embedQuery(question).then(function (q) {
      // Stages branch on q.vec, never on configuration: a per-query embed failure degrades identically.
      var params = matchParameters(question, q.vec);
      var evidence = scoreEvidence(question, q.vec, params.cited);
      var picked = selectChunks(evidence);
      return retrievalResult(q, params, evidence, picked);
    });
  }

  function capText(text, cap) {
    if (cap <= 0 || text.length <= cap) return text;
    return text.slice(0, cap).replace(/\s+\S*$/, "") + " …";
  }

  // Classify from the index: source_id prefixes left 254 chunks untyped and read XML samples as prose.
  var EVIDENCE_CLASS_BY_KIND = {
    paper: "prose", website: "prose", markdown: "prose", config: "sample", code: "code"
  };
  // `repo` is a grab-bag of all three: a README, a utility module, and a sample config.
  var EVIDENCE_CLASS_BY_SOURCE = {
    "repo.readme": "prose", "repo.gpu_info": "code", "repo.sample_config": "sample"
  };

  // Unknown kinds fall to `code`: least authority for the three gated classes, valid grounding everywhere else.
  function evidenceClass(sourceId) {
    var byId = EVIDENCE_CLASS_BY_SOURCE[sourceId];
    if (byId) return byId;
    var src = (knowledgeIndex && knowledgeIndex.sources && knowledgeIndex.sources[sourceId]) || {};
    return EVIDENCE_CLASS_BY_KIND[src.kind] || "code";
  }

  function hasProseEvidence(chunks) {
    return chunks.some(function (c) { return evidenceClass(c.source_id) === "prose"; });
  }

  function evidenceContext(chunks) {
    if (!chunks.length) return "- none";
    return chunks.map(function (c) {
      return "[" + evidenceClass(c.source_id) + " | " + c.source_id + " " + c.heading + "] " +
        capText(c.text, EVIDENCE_CHAR_CAP);
    }).join("\n\n");
  }

  // Restated below the evidence so the deciding rule is read last; only the question class is left to the model.
  function operativeRule(chunks) {
    if (!chunks.length) return "";
    return "\n\nDecision reminder for this turn: " + (hasProseEvidence(chunks)
      ? "`prose` evidence IS present above, so the source-class gate does not apply this turn. " +
        "The general rule governs: answer if an item supports the fact, refuse if none does. " +
        "Prose being present is not itself support -- it may address a different topic."
      : "NO `prose` evidence was retrieved -- every item above is `sample` or `code`. If the " +
        "question asks about UI behavior, a recommended or default value, or the runtime " +
        "semantics of a configurator toggle, reply 'INSUFFICIENT_EVIDENCE: <one-sentence " +
        "reason>'. For any other question class -- definitions, what fields/keys exist, code " +
        "structure, numbers stated in the text -- answer normally from the items above.");
  }

  function parameterContext(specs) {
    if (!specs.length) return "- none";
    return specs.map(function (p) {
      return "- " + p.label + " (" + p.xml_path + "): " + p.summary + " " + p.significance;
    }).join("\n");
  }

  function historyTranscript() {
    return history.slice(-HISTORY_TURNS).map(function (t) {
      return "User: " + t.q + "\nAssistant: " + t.a;
    }).join("\n");
  }

  function historyContext() { return history.length ? historyTranscript() : "- none"; }

  // Last turn split out so recency is structural: a similar OLD turn used to win the referent.
  function condenseTranscript() {
    var turns = history.slice(-HISTORY_TURNS);
    var last = turns[turns.length - 1];
    var earlier = turns.slice(0, -1).map(function (t) {
      return "User: " + t.q + "\nAssistant: " + t.a;
    }).join("\n");
    return (earlier ? "Earlier turns (background only, not the referent):\n" + earlier + "\n\n" : "") +
      "MOST RECENT turn -- the referent for anything the latest question leaves unsaid:\n" +
      "User: " + last.q + "\nAssistant: " + last.a;
  }

  function recordTurn(q, a) {
    a = (a || "").split(/\s+/).join(" ").trim();
    if (a.length > HISTORY_ANSWER_CHAR_CAP) a = a.slice(0, HISTORY_ANSWER_CHAR_CAP).replace(/\s+\S*$/, "") + " …";
    history.push({ q: (q || "").trim(), a: a });
    if (history.length > HISTORY_TURNS) history = history.slice(-HISTORY_TURNS);
  }

  // History feeds genuine follow-ups only; the two directives exclude each other by construction.
  function buildHumanPrompt(question, retrieval, isFollowup, allowClarify, valueAnswer) {
    return "Conversation history (reference resolution only; NOT evidence):\n" +
      (isFollowup ? historyContext() : "- none") + "\n\n" +
      "Question: " + question + "\n\n" +
      "Matched parameters:\n" + parameterContext(retrieval.specs) + "\n\n" +
      "Evidence (each item is `[class | source_id heading_path] text`):\n" + evidenceContext(retrieval.chunks) +
      operativeRule(retrieval.chunks) +
      (valueAnswer ? VALUE_ANSWER_DIRECTIVE : (allowClarify ? CLARIFY_DIRECTIVE : ""));
  }

  // One-shot abort timer; deliberately never cleared -- aborting a settled request is a no-op.
  function abortAfter(ms) {
    var c = new AbortController();
    setTimeout(function () { c.abort(); }, ms);
    return c;
  }

  // Parse an Ollama parameter_size string (e.g. '7.6B') to billions.
  function parseParamsB(size) {
    var m = /^([\d.]+)\s*([MB])$/.exec(String(size || "").trim().toUpperCase());
    if (!m) return null;
    var v = parseFloat(m[1]);
    return m[2] === "M" ? v / 1000 : v;
  }

  function modelParamsB(model) {
    return fetch(OLLAMA + "/api/show", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model }),
      signal: abortAfter(5000).signal
    }).then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) { return j ? parseParamsB(j.details && j.details.parameter_size) : null; })
      .catch(function () { return null; });
  }

  // Also stores the split into chatModels/embedModels, which populateModelSelect and connect read afterwards.
  function probeOllama() {
    return fetch(OLLAMA + "/api/tags", { signal: abortAfter(3500).signal })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (j) {
        var all = (j.models || []).map(function (m) { return m.name; });
        modelDigests = {};
        (j.models || []).forEach(function (m) { modelDigests[m.name] = m.digest || ""; });
        var isEmbed = function (n) { return /embed|bge|minilm|nomic|mxbai|arctic|gte|\be5\b/i.test(n); };
        chatModels = all.filter(function (n) { return !isEmbed(n); });
        embedModels = all.filter(isEmbed);
        return { ok: true, models: chatModels, embedModels: embedModels };
      })
      .catch(function (e) { return { ok: false, error: e.message || String(e) }; });
  }

  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function chatComplete(messages, attempt) {
    attempt = attempt || 0;
    return fetch(OLLAMA + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: chatModel, messages: messages, stream: false, think: false, options: { temperature: 0 } }),
      signal: abortAfter(LLM_TIMEOUT_MS).signal
    }).then(function (r) {
      if (!r.ok) {
        var httpError = new Error("Ollama HTTP " + r.status);
        httpError.status = r.status;
        throw httpError;
      }
      return r.json();
    }).then(function (j) { return (j.message && j.message.content ? j.message.content : "").trim(); })
      .catch(function (e) {
        // 4xx is permanent (model removed) and an abort already spent the full timeout: retrying either just multiplies the wait.
        var permanent = !!e && ((e.status >= 400 && e.status < 500) || e.name === "AbortError");
        if (!permanent && attempt < LLM_RETRIES - 1) return delay(1500 * (attempt + 1)).then(function () { return chatComplete(messages, attempt + 1); });
        throw e;
      });
  }

  function normalizeQuestion(s) { return (s || "").toLowerCase().split(/\s+/).join(" ").replace(/^[\s?.!]+|[\s?.!]+$/g, ""); }

  // Fold a follow-up's referents into a standalone question before retrieval.
  function contextualize(question) {
    if (!history.length) return Promise.resolve(question);
    var hist = condenseTranscript();
    var messages = [
      { role: "system", content: CONDENSE_SYSTEM },
      { role: "user", content: "Conversation:\n" + hist + "\n\nLatest question: " + question + "\n\nStandalone question:" }
    ];
    return chatComplete(messages).then(function (text) {
      return (text || "").replace(/^["'`]+|["'`]+$/g, "").trim() || question;
    }).catch(function () { return question; });
  }

  function needsGoal(question) {
    return chatComplete([
      { role: "system", content: NEEDS_GOAL_SYSTEM },
      { role: "user", content: question }
    ]).then(function (t) { return (t || "").trim().toUpperCase().indexOf("YES") === 0; })
      .catch(function () { return false; });
  }

  // Engage a no-evidence turn conversationally, or decline honestly.
  function engageWithoutEvidence(question) {
    return chatComplete([
      { role: "system", content: NO_EVIDENCE_TURN_SYSTEM },
      { role: "user", content: "Conversation so far:\n" + historyContext() + "\n\nUser's latest message: " + question }
    ]).then(function (t) { return (t || "").trim() || INTERACTIVE_NO_EVIDENCE_MESSAGE; })
      .catch(function () { return INTERACTIVE_NO_EVIDENCE_MESSAGE; });
  }

  // onPartial gets the WHOLE answer accumulated so far on every chunk, not the newest delta.
  function streamChat(question, retrieval, onPartial, isFollowup, allowClarify, valueAnswer) {
    var body = {
      model: chatModel,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: buildHumanPrompt(question, retrieval, isFollowup, allowClarify, valueAnswer) }
      ],
      stream: true,
      think: false,  // = Python reasoning_effort=none
      options: { temperature: 0 }
    };
    // Idle timeout re-armed per chunk: kill a stalled stream, never a slow-but-advancing CPU one.
    var ctl = new AbortController();
    var idle;
    function arm(ms) { idle = setTimeout(function () { ctl.abort(); }, ms); }
    function disarm() { clearTimeout(idle); }
    // No bytes flow during connect, model load and prompt eval, so the first window gets the full LLM budget.
    arm(LLM_TIMEOUT_MS);
    return fetch(OLLAMA + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctl.signal
    }).then(function (resp) {
      if (!resp.ok) { disarm(); throw new Error("Ollama HTTP " + resp.status); }
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buf = "";
      var full = "";
      function pump() {
        return reader.read().then(function (res) {
          disarm();
          if (res.done) return full;
          buf += decoder.decode(res.value, { stream: true });
          var lines = buf.split("\n");
          buf = lines.pop();
          for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line) continue;
            var obj;
            try { obj = JSON.parse(line); } catch (e) { continue; }
            if (obj.message && obj.message.content) {
              full += obj.message.content;
              onPartial(full);
            }
          }
          arm(STREAM_IDLE_MS);
          return pump();
        });
      }
      // A stall mid-answer surrenders what already streamed rather than wiping it with an error.
      return pump().catch(function (e) {
        disarm();
        if (full) return full + " …";
        throw e;
      });
    });
  }

  // VERIFICATION_PHRASES match the whole normalized message; PROVENANCE_STEMS only its prefix.
  var VERIFICATION_PHRASES = {};
  ("you sure|are you sure|sure|really|seriously|for real|is that right|is that correct|is that true|" +
   "is that accurate|are you certain|you positive|that doesn't sound right|i don't believe you|" +
   "doubt that|doubt it").split("|").forEach(function (p) { VERIFICATION_PHRASES[p] = 1; });
  var PROVENANCE_STEMS = ["how do you know", "how can you be sure", "how would you know", "says who",
    "prove it", "back that up", "where does it say", "what is your source", "what's your source",
    "according to what", "based on what", "why should i believe", "why should i trust"];
  var ANAPHORA = {};
  "that this it these those so then same".split(" ").forEach(function (w) { ANAPHORA[w] = 1; });
  var GENERIC_TOKENS = {};
  ("gem gemprf prf prfs model models value values number set setting settings parameter parameters " +
   "config configuration analysis default choose use what how which does your the for").split(" ")
    .forEach(function (w) { GENERIC_TOKENS[w] = 1; });

  function normalizePhrase(text) {
    return (text || "").trim().replace(/^["']+|["']+$/g, "").replace(/[?!.]+$/, "")
      .toLowerCase().split(/\s+/).join(" ").trim();
  }
  function contentTokens(text) {
    var out = {};
    (String(text || "").toLowerCase().match(/[a-z0-9_]+/g) || []).forEach(function (t) {
      if (t.length > 2 && !GENERIC_TOKENS[t]) out[t] = 1;
    });
    return out;
  }
  // Detect a contentless doubt/provenance follow-up ('you sure?').
  function isVerificationFollowup(question) {
    if (!history.length) return false;
    var norm = normalizePhrase(question);
    if (!norm || norm.split(" ").length > 10) return false;
    if (VERIFICATION_PHRASES[norm]) return true;
    for (var i = 0; i < PROVENANCE_STEMS.length; i++) {
      var stem = PROVENANCE_STEMS[i];
      if (norm === stem || norm.indexOf(stem + " ") === 0) {
        var rest = contentTokens(norm.slice(stem.length));
        if (!Object.keys(rest).some(function (t) { return !ANAPHORA[t]; })) return true;
      }
    }
    return false;
  }
  function firstToken(text) {
    var tok = (text || "").toUpperCase().trim().split(/\s+/)[0] || "";
    return tok.replace(/^[.,!?:;"'`()]+|[.,!?:;"'`()]+$/g, "");
  }

  // buildHumanPrompt without isFollowup/allowClarify on purpose: the re-check stands on this retrieval alone.
  function generateAnswer(question, retrieval) {
    return chatComplete([
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: buildHumanPrompt(question, retrieval) }
    ]).then(function (raw) {
      if (isRefusal(raw)) return { answer: raw, refused: true };
      return { answer: raw, refused: false };
    });
  }

  // Fresh retrieve + non-streaming answer for the verify path; answered=false means INSUFFICIENT_EVIDENCE.
  function retrieveAndAnswer(query) {
    return retrieve(query).then(function (retrieval) {
      return generateAnswer(query, retrieval).then(function (g) {
        return { ret: retrieval, answer: g.answer, answered: !g.refused };
      });
    });
  }

  var ADJUDICATE_SYSTEM =
    "You check whether newly retrieved GEM-pRF evidence still supports an earlier answer. You are " +
    "given the EARLIER answer and a FRESH answer generated from a new retrieval of the same question. " +
    "Reply with exactly one word: CONFIRMS (the fresh answer agrees on the key fact), CORRECTS (the " +
    "fresh answer contradicts or materially changes the key fact), or UNSUPPORTED (the fresh answer " +
    "does not actually establish the earlier claim). Judge only the key factual claim, not wording.";
  var ADJUDICATE_VERDICTS = { CONFIRMS: 1, CORRECTS: 1, UNSUPPORTED: 1 };
  var VERIFY_CAVEAT =
    "\n\n(That's what the GEM-pRF sources document — a stated default/behavior, not a recommendation " +
    "tuned to your data or goal.)";

  function firstSentence(text) {
    var s = String(text || "").trim();
    var m = /[.;]\s/.exec(s);
    return m ? s.slice(0, m.index + 1) : s;
  }

  function adjudicateConsistency(prior, fresh) {
    return chatComplete([
      { role: "system", content: ADJUDICATE_SYSTEM },
      { role: "user", content: "EARLIER answer:\n" + prior.slice(0, 1500) + "\n\nFRESH answer:\n" + fresh.slice(0, 1500) + "\n\nOne word:" }
    ]).then(function (t) { var tok = firstToken(t); return ADJUDICATE_VERDICTS[tok] ? tok : ""; })
      .catch(function () { return ""; });
  }

  function composeVerify(verdict, body) {
    body = (body || "").trim();
    if (verdict === "CONFIRMS") return "Yes — re-checking the sources, that holds up:\n\n" + body + VERIFY_CAVEAT;
    if (verdict === "CORRECTS") return "Actually, let me correct that. Re-checking the sources:\n\n" + body + VERIFY_CAVEAT;
    return "Here is what the GEM-pRF sources actually say when I re-check:\n\n" + body + VERIFY_CAVEAT;
  }

  // Confirm-or-correct the prior answer from a fresh retrieval.
  function verifyPriorAnswer() {
    var prior = history[history.length - 1];
    if (!prior || !(prior.a || "").trim()) return Promise.resolve(false);
    // The re-query carries the prior answer's first sentence, so retrieval lands on the doubted claim.
    var recheckQuery = (prior.q + " " + firstSentence(prior.a)).trim();
    return retrieveAndAnswer(recheckQuery).then(function (recheck) {
      if (!recheck.answered) {
        fillBubble("I can't re-confirm that from the allowed GEM-pRF sources right now — " +
          "re-checking, I don't find support for it.");
        return true;
      }
      return adjudicateConsistency(prior.a, recheck.answer).then(function (verdict) {
        fillBubble(composeVerify(verdict, recheck.answer));
        renderCitations(activeBot, recheck.ret.chunks);
        return true;
      });
    });
  }

  function fillBubble(markdown) {
    activeBot.classList.remove("gpa-cursor");
    activeBot.innerHTML = renderMarkdown(markdown);
    scrollDown();
  }

  // html goes in raw as innerHTML -- every caller must escape first; that is why addUserMsg uses textContent.
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function scrollDown() { els.messages.scrollTop = els.messages.scrollHeight; }

  function addUserMsg(text) {
    var m = el("div", "gpa-msg gpa-user");
    m.textContent = text;
    els.messages.appendChild(m);
    scrollDown();
  }

  function addBotMsg() {
    var m = el("div", "gpa-msg gpa-bot gpa-cursor", "");
    els.messages.appendChild(m);
    scrollDown();
    return m;
  }

  function setStatus(state) {
    els.dot.className = "gpa-status-dot" + (state === "ok" ? " gpa-ok" : state === "busy" ? " gpa-busy" : "");
  }

  function renderCitations(node, chunks) {
    if (!chunks.length) return;
    var seen = Object.create(null);
    var items = [];
    chunks.forEach(function (c) {
      var src = (knowledgeIndex.sources && knowledgeIndex.sources[c.source_id]) || {};
      var key = c.source_id;
      if (seen[key]) return;
      seen[key] = 1;
      var title = src.title || c.source_id;
      var url = src.url || "";
      items.push(url
        ? '<span class="gpa-cite-item">• <a href="' + escapeHtml(url) + '" target="_blank" rel="noopener">' + escapeHtml(title) + "</a></span>"
        : '<span class="gpa-cite-item">• ' + escapeHtml(title) + "</span>");
    });
    var box = el("div", "gpa-cites", "<strong>Sources</strong>" + items.join(""));
    node.appendChild(box);
  }

  // Pend a single-reply clarify; `fork` is kept so the reply can bind back to a candidate.
  function askClarify(originalQuestion, clarifyingQuestion, isValueChoice, fork, reAsked) {
    // `asked` is kept because a reply means nothing alone: '8' has no unit once the bubble scrolls away.
    pendingClarify = {
      question: originalQuestion, asked: clarifyingQuestion,
      // reAsked caps the fork re-ask at one, so an unresolvable reply cannot loop the user forever.
      valueChoice: !!isValueChoice, fork: fork || [], reAsked: !!reAsked
    };
    fillBubble(clarifyingQuestion);
  }

  var CODE_EXPLAIN_SYSTEM =
    "The user asked to see GEM-pRF source code, and the exact code (one or more snippets) has ALREADY been " +
    "shown to them above. In 1-4 plain-English sentences, explain what it does; when several snippets are " +
    "shown, say how they relate (e.g. where a value is parsed, passed, and used). Do NOT reproduce, rewrite, " +
    "quote, or include any code, identifiers-as-code, or pseudocode -- it is already displayed. Describe only " +
    "behaviour and purpose.";

  function isCodeRequest(q) {
    return /\bcode\b|\bimplementation\b|\bsource[- ]?code\b/i.test(q || "");
  }

  // Longest matching probe wins, so a multi-word label beats a short alias another parameter also carries.
  function codeParamForQuestion(question) {
    var q = " " + String(question || "").toLowerCase() + " ";
    var qSpaces = q.replace(/[_.]/g, " ");
    var best = null, bestLen = 0;
    (knowledgeIndex.parameters || []).forEach(function (p) {
      var probes = p.id.toLowerCase().split(".").concat(p.id.toLowerCase(), (p.label || "").toLowerCase());
      (p.aliases || []).forEach(function (a) { probes.push((a || "").toLowerCase()); });
      probes.forEach(function (probe) {
        probe = probe.trim();
        if (probe.length < 3) return;  // 2-char terms match inside common words
        var flat = probe.replace(/[_.]/g, " ");
        var re = new RegExp("(^|[^a-z0-9])" + probe.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "([^a-z0-9]|$)");
        if ((re.test(q) || qSpaces.indexOf(" " + flat + " ") !== -1) && probe.length > bestLen) { bestLen = probe.length; best = p; }
      });
    });
    return best;
  }

  // Generic code tokens too common to locate a parameter's real usage lines.
  var GENERIC_CODE = {};
  "value values true false none str int float default self node key enable config param params get set data".split(" ").forEach(function (w) { GENERIC_CODE[w] = 1; });

  // Prefers the quoted config-key line (the real usage) over bare mentions.
  function focusExcerpt(text, terms) {
    var distinctive = terms.filter(function (w) { return w.length >= 3 && !GENERIC_CODE[w]; });
    if (!distinctive.length) return text;
    var lines = text.split("\n");
    function hitsFor(needles) {
      var h = [];
      lines.forEach(function (ln, i) { var low = ln.toLowerCase(); for (var k = 0; k < needles.length; k++) { if (low.indexOf(needles[k]) !== -1) { h.push(i); break; } } });
      return h;
    }
    var quoted = [];
    distinctive.forEach(function (w) { quoted.push('"' + w + '"'); quoted.push("'" + w + "'"); });
    var hitLines = hitsFor(quoted);            // the config-key line, if any
    if (!hitLines.length) hitLines = hitsFor(distinctive);  // else any mention
    if (!hitLines.length) return text;
    var CTX = 4;
    return lines.slice(Math.max(0, hitLines[0] - CTX), Math.min(lines.length, hitLines[hitLines.length - 1] + CTX + 1)).join("\n");
  }

  // True when a chunk reads the parameter as a quoted config key -- then even the primary block is trimmed.
  function isConfigReadChunk(text, terms) {
    var low = String(text || "").toLowerCase();
    var distinctive = terms.filter(function (w) { return w.length >= 3 && !GENERIC_CODE[w]; });
    for (var k = 0; k < distinctive.length; k++) { if (low.indexOf('"' + distinctive[k] + '"') !== -1 || low.indexOf("'" + distinctive[k] + "'") !== -1) return true; }
    return false;
  }

  // Best chunk per code source, most param-dense first; falls back to the best retrieved code chunk.
  function findCodeChunks(question, retrieval) {
    var specs = retrieval.specs || [];
    var namedSpec = codeParamForQuestion(question) || specs[0];
    var isCode = function (c) {
      var s = knowledgeIndex.sources[c.source_id];
      return s && s.kind === "code" && !/tests?[._/]|_test|test_/i.test(c.source_id);
    };
    var terms = namedSpec ? tokenize([namedSpec.label].concat(namedSpec.aliases || []).join(" ")) : tokenize(question);
    function termDensity(c) {
      var t = c.text.toLowerCase(), s = 0;
      terms.forEach(function (w) { var i = 0, n = 0; while ((i = t.indexOf(w, i)) !== -1) { n++; i += w.length; } s += n; });
      if (/(^|\n)\s*(def|class)\s/.test(c.text)) s += 0.5;
      return s;
    }
    var distinctive = terms.filter(function (w) { return w.length >= 3 && !GENERIC_CODE[w]; });
    // The parameter as a quoted config key marks the real usage site, which a bare-word count misses.
    function configBoost(c) {
      var low = c.text.toLowerCase();
      for (var i = 0; i < distinctive.length; i++) {
        if (low.indexOf('"' + distinctive[i] + '"') !== -1 || low.indexOf("'" + distinctive[i] + "'") !== -1) return 1000;
      }
      return 0;
    }
    // Pick within a source by base+boost (fixes an overloaded term), but order sources by base.
    function bestIn(chunks) {
      var best = null, bestBoosted = -1, bestBase = -1;
      chunks.forEach(function (c) { var base = termDensity(c), b = base + configBoost(c); if (b > bestBoosted) { best = c; bestBoosted = b; bestBase = base; } });
      return { chunk: best, base: bestBase, boosted: bestBoosted };
    }
    var sids = namedSpec ? (namedSpec.code_source_ids || []) : [];
    var picked = [];
    sids.forEach(function (sid) {
      var chunks = knowledgeIndex.chunks.filter(function (c) { return c.source_id === sid && isCode(c); });
      if (!chunks.length) return;
      var bestForSource = bestIn(chunks);
      if (bestForSource.chunk && bestForSource.boosted > 0) picked.push(bestForSource);  // only sources where the parameter actually appears
    });
    if (picked.length) {
      picked.sort(function (a, b) { return b.base - a.base; });  // implementation (most param-dense) leads
      return picked.slice(0, 5).map(function (p) { return p.chunk; });
    }
    var pool = retrieval.chunks.filter(isCode);  // no curated match: best retrieved code chunk, or nothing
    var f = bestIn(pool);
    return f.chunk ? [f.chunk] : [];
  }

  function findCodeChunk(question, retrieval) {
    var arr = findCodeChunks(question, retrieval);
    return arr.length ? arr[0] : null;
  }

  // Show code verbatim: the model rewrites real code (wrong signatures, dropped lines) when asked to quote it.
  function showCodeVerbatim(question, retrieval, codeChunks) {
    var namedSpec = codeParamForQuestion(question) || (retrieval.specs && retrieval.specs[0]);
    var terms = namedSpec ? tokenize([namedSpec.label].concat(namedSpec.aliases || []).join(" ")) : tokenize(question);
    var blocks = codeChunks.map(function (c, i) {
      // Strips the indexer's breadcrumb line; the same pattern can bite a code line containing ' > '.
      var raw = String(c.text || "").replace(/^[^\n]{1,160} > [^\n]{0,160}\n+/, "").trim();
      // Full text only for a primary that's an actual implementation; a config-read or secondary site is trimmed.
      var code = (i === 0 && !isConfigReadChunk(raw, terms)) ? raw : focusExcerpt(raw, terms);
      var title = (knowledgeIndex.sources[c.source_id] && knowledgeIndex.sources[c.source_id].title) || c.source_id;
      return { title: title, code: code, src: c.source_id };
    });
    var md = blocks.map(function (b) { return "**" + b.title + "**\n\n```python\n" + b.code + "\n```"; }).join("\n\n");
    var srcList = blocks.map(function (b) { return b.src; }).join(", ");
    activeBot.classList.remove("gpa-cursor");
    activeBot.innerHTML = renderMarkdown(md + "\n\n_explaining…_");
    scrollDown();
    var explainUser = "Question: " + question + "\n\nThe code shown:\n" +
      blocks.map(function (b) { return "From " + b.src + ":\n" + b.code; }).join("\n\n") +
      "\n\nYour explanation (no code):";
    return chatComplete([
      { role: "system", content: CODE_EXPLAIN_SYSTEM },
      { role: "user", content: explainUser }
    ]).then(function (expl) {
      activeBot.innerHTML = renderMarkdown(md + "\n\n" + ((expl || "").trim()));
      renderCitations(activeBot, codeChunks.concat(retrieval.chunks));
      recordTurn(question, "Showed source code from " + srcList + ". " + (expl || "").trim());
      scrollDown();
    }).catch(function () {
      activeBot.innerHTML = renderMarkdown(md);
      renderCitations(activeBot, codeChunks.concat(retrieval.chunks));
      recordTurn(question, "Showed source code from " + srcList + ".");
      scrollDown();
    });
  }

  // Normal answer path: stream, refuse honestly, or ask one CLARIFY question.
  function answerNormally(question, retrieval, standalone, allowClarify, valueAnswer) {
    var refusalMsg = allowClarify ? INTERACTIVE_NO_EVIDENCE_MESSAGE : INSUFFICIENT_EVIDENCE_MESSAGE;
    if (!retrieval.chunks.length) {
      var noEv = allowClarify ? engageWithoutEvidence(question) : Promise.resolve(INSUFFICIENT_EVIDENCE_MESSAGE);
      return noEv.then(function (text) { fillBubble(text); recordTurn(question, text); });
    }
    // Detect and locate code from the condensed question, but explain using the user's literal wording.
    if (isCodeRequest(standalone || question)) {
      var codeChunks = findCodeChunks(standalone || question, retrieval);
      if (codeChunks.length) {
        return showCodeVerbatim(question, retrieval, codeChunks);
      }
    }
    var isFollowup = !!standalone && normalizeQuestion(standalone) !== normalizeQuestion(question);
    var streamedText = "";
    // Anchored like isRefusal, surviving a leading quote or ** -- models wrap this sentinel the same way.
    function isClarify(t) { return allowClarify && /^["'`*\s]*CLARIFY:/i.test((t || "").trim()); }
    function clarifyText(t) { return (t || "").trim().replace(/^["'`*\s]*CLARIFY:\s*/i, "").replace(/["'`*\s]+$/, "").trim(); }
    return streamChat(question, retrieval, function (full) {
      streamedText = full;
      if (isClarify(full)) activeBot.innerHTML = renderMarkdown(clarifyText(full));
      // The sentinel opens the reply, so this fires on the first chunk; re-decided from the whole text below.
      else if (isRefusal(full)) activeBot.innerHTML = renderMarkdown(refusalMsg);
      else activeBot.innerHTML = renderMarkdown(full);
      scrollDown();
    }, isFollowup, allowClarify, valueAnswer).then(function () {
      activeBot.classList.remove("gpa-cursor");
      if (isClarify(streamedText)) {
        var clarifyQuestion = clarifyText(streamedText) || "Could you say a bit more about what you're trying to do?";
        askClarify(question, clarifyQuestion);
        return;
      }
      var refused = isRefusal(streamedText);
      var answer = refused ? refusalMsg : streamedText.trim();
      activeBot.innerHTML = renderMarkdown(answer);
      if (!refused) renderCitations(activeBot, retrieval.chunks);
      recordTurn(question, answer);
      scrollDown();
    });
  }

  // Core turn: condense, retrieve, then route to value-clarify, fork-clarify, or answer.
  function corePipeline(question, allowClarify, valueAnswer) {
    return contextualize(question).then(function (condensed) {
      var standalone = (normalizeQuestion(question) !== normalizeQuestion(condensed)) ? condensed : question;
      var isFollowup = normalizeQuestion(standalone) !== normalizeQuestion(question);
      // valueChoiceActive outlives the clarify reply; a question that is not a follow-up ends it.
      if (allowClarify && valueChoiceActive && !isFollowup) valueChoiceActive = false;
      var continuingValueChoice = allowClarify && valueChoiceActive && isFollowup;
      var goalCheck = (allowClarify && !continuingValueChoice) ? needsGoal(condensed) : Promise.resolve(false);
      return Promise.all([retrieve(condensed), goalCheck]).then(function (r) {
        var retrieval = r[0], mustAskGoal = r[1];
        var clarifying = allowClarify && !continuingValueChoice;
        // A term matching a setting is not the question being ABOUT it; may resolve to empty.
        var forkGate = Promise.resolve([]);
        if (clarifying) {
          var seeded = forkCandidates(condensed, retrieval.forkPool);
          forkGate = forkIsTheSubject(condensed, seeded).then(function (aboutTheSettings) {
            return (seeded.length && !aboutTheSettings) ? [] : seeded;
          });
        }
        return forkGate.then(function (fork) {
          if (clarifying && mustAskGoal) {
            valueChoiceActive = true;
            // fork rides along: generateValueClarify prepends the fork question, so the reply answers it too.
            return generateValueClarify(condensed, retrieval, fork).then(function (clarifyQuestion) { askClarify(question, clarifyQuestion, true, fork); });
          }
          if (clarifying && fork.length) {
            askClarify(question, forkQuestion(fork), false, fork);
            return;
          }
          return answerNormally(question, retrieval, standalone, clarifying, valueAnswer);
        });
      });
    });
  }

  // Verify gate: a contentless 'you sure?' re-checks the last answer; everything else goes to corePipeline.
  function processQuestion(question) {
    if (isVerificationFollowup(question)) {
      return verifyPriorAnswer().then(function (handled) {
        if (handled) return;
        return corePipeline(question, true);
      });
    }
    return corePipeline(question, true);
  }

  // Handle a user submit, or a reply to a pending clarify.
  function ask(question) {
    // indexing blocks the Enter key too, which the disabled button alone would not.
    if (busy || !connected || indexing) return;
    question = question.trim();
    if (!question && !pendingClarify) return;
    busy = true;
    setStatus("busy");
    els.send.disabled = true;
    if (question) addUserMsg(question);
    els.input.value = "";
    els.input.style.height = "auto";
    activeBot = addBotMsg();

    function fail(e) {
      activeBot.classList.remove("gpa-cursor");
      activeBot.innerHTML = renderMarkdown("**Generation failed:** " + (e.message || e) +
        "\n\nMake sure Ollama is running and the model is available.");
      pendingClarify = null;
    }
    function done() {
      busy = false;
      setStatus(connected ? "ok" : "off");
      els.send.disabled = false;
      els.input.focus();
    }

    // A clarify reply is folded into the ORIGINAL question and re-run with clarifying disabled.
    if (pendingClarify) {
      var original = pendingClarify.question;
      var valueChoice = pendingClarify.valueChoice;
      var fork = pendingClarify.fork;
      var asked = pendingClarify.asked;
      var reAsked = pendingClarify.reAsked;
      pendingClarify = null;
      var keep = question && !/^(quit|exit|cancel|stop)$/i.test(question);
      // A cancelled or blank reply gave no goal, so the value directive's given-goal premise must not run.
      if (!keep) valueChoiceActive = false;
      var picked = keep ? resolveForkReply(question, fork) : null;
      // Unreadable reply: re-ask rather than let the ranking pick. Value clarifies reply with a goal, so are exempt.
      if (keep && !picked && !valueChoice && !reAsked && fork && fork.length >= 2) {
        askClarify(original, forkReaskQuestion(fork, question), false, fork, true);
        done();
        return;
      }
      var folded = keep ? foldClarifyReply(original, question, fork, asked) : original;
      corePipeline(folded, false, keep && valueChoice).catch(fail).then(done);
      return;
    }

    processQuestion(question).catch(fail).then(done);
  }

  // Only one banner lives in the message area at a time -- this drops the previous one.
  function showBanner(html, isError) {
    var existing = els.messages.querySelector(".gpa-banner");
    if (existing) existing.remove();
    var b = el("div", "gpa-banner" + (isError ? " gpa-error" : ""), html);
    els.messages.appendChild(b);
    scrollDown();
    return b;
  }

  // Assembled in JS so the OLLAMA_ORIGINS command shows the visitor's own origin.
  function setupInstructionsHtml() {
    var origin = window.location.origin;
    return "<strong>Connect your local model</strong><br>" +
      "This assistant runs the LLM on <em>your</em> machine — nothing is sent to any server. " +
      "One-time setup:" +
      "<br>1. Install <a href=\"https://ollama.com/download\" target=\"_blank\" rel=\"noopener\">Ollama</a> " +
      "and pull the model:" +
      "<code>ollama pull " + DEFAULT_MODEL + "</code>" +
      "2. Pull the embedding model (required for retrieval):" +
      "<code>ollama pull " + PREFERRED_EMBED + "</code>" +
      "3. Allow this site to reach Ollama — run the line for your system, then restart Ollama:" +
      "<br>macOS:" +
      "<code>launchctl setenv OLLAMA_ORIGINS \"" + origin + "\"</code>" +
      "Linux, or when starting manually:" +
      "<code>OLLAMA_ORIGINS=\"" + origin + "\" ollama serve</code>" +
      "<button class=\"gpa-btn gpa-secondary\" id=\"gpa-retry\">Retry connection</button>" +
      "<div class=\"gpa-disclaimer\">Note: needs a Chromium-based browser (Chrome/Edge/Brave). Safari blocks localhost requests from HTTPS pages.</div>";
  }

  function populateModelSelect() {
    els.modelSelect.innerHTML = "";
    chatModels.forEach(function (n) {
      var o = el("option");
      o.value = n; o.textContent = n;
      els.modelSelect.appendChild(o);
    });
    // Also (re)sets chatModel: DEFAULT_MODEL when installed, else the first listed.
    chatModel = chatModels.indexOf(DEFAULT_MODEL) !== -1 ? DEFAULT_MODEL : chatModels[0];
    els.modelSelect.value = chatModel;
  }

  function removeWeakBanner() {
    var w = els.messages.querySelector(".gpa-weakwarn");
    if (w) w.remove();
  }

  function warnIfWeakModel() {
    removeWeakBanner();
    if (!connected) return;
    // Snapshot the model: /api/show is async, so a switch mid-flight discards the result.
    var model = chatModel;
    modelParamsB(model).then(function (paramsB) {
      if (paramsB === null || paramsB >= MIN_MODEL_PARAMS_B || chatModel !== model) return;
      var w = el("div", "gpa-banner gpa-weakwarn",
        "<strong>" + escapeHtml(model) + "</strong> has only " + paramsB.toFixed(1) + "B parameters — likely not " +
        "strong enough for reliable grounded answers. Prefer a " + MIN_MODEL_PARAMS_B.toFixed(0) + "B+ model (e.g. " + DEFAULT_MODEL + ").");
      els.messages.appendChild(w);
      scrollDown();
    });
  }

  // Hold the input until the corpus is embedded: questions asked before that retrieve BM25-only and cannot fork.
  function setIndexing(on) {
    indexing = on;
    // Its own class, not showBanner: that one drops any .gpa-banner, including the weak-model warning.
    var existing = els.messages.querySelector(".gpa-indexing");
    if (on && !existing) {
      els.messages.appendChild(el("div", "gpa-banner gpa-indexing",
        "Indexing documentation… the assistant answers once the search index is built. " +
        "This runs once — later visits reuse the cached index."));
      scrollDown();
    } else if (!on && existing) {
      existing.remove();
    }
    els.input.disabled = on;
    els.input.placeholder = on ? "Indexing documentation…" : INPUT_PLACEHOLDER;
    // Never enable Send under a turn already in flight -- ask()'s done() owns the button then.
    els.send.disabled = on || busy;
    if (!on) els.input.focus();
  }

  // Connect to Ollama: probe, greet, size-gate, then build vectors.
  function connect() {
    var b = showBanner("Connecting to your local Ollama…", false);
    return probeOllama().then(function (probeResult) {
      if (!probeResult.ok) {
        connected = false;
        setStatus("off");
        var banner = showBanner(setupInstructionsHtml(), true);
        var retry = banner.querySelector("#gpa-retry");
        if (retry) retry.addEventListener("click", connect);
        return;
      }
      // Both roles are checked up front so a fresh visitor sees every missing pull in one banner.
      embedModel = pickEmbedModel(embedModels);
      chunkVectors = null;
      paramRows = null;
      var missing = [];
      if (!probeResult.models.length) missing.push("a chat model (writes the answers)<code>ollama pull " + DEFAULT_MODEL + "</code>");
      if (!embedModel) missing.push("an embedding model (powers retrieval)<code>ollama pull " + PREFERRED_EMBED + "</code>");
      if (missing.length) {
        connected = false;
        setStatus("off");
        var missingBanner = showBanner("Ollama is running but the assistant still needs " + missing.join(" and ") +
          "<button class=\"gpa-btn gpa-secondary\" id=\"gpa-retry\">Retry</button>", true);
        var missingRetry = missingBanner.querySelector("#gpa-retry");
        if (missingRetry) missingRetry.addEventListener("click", connect);
        return;
      }
      connected = true;
      setStatus("ok");
      populateModelSelect();
      var staleBanner = els.messages.querySelector(".gpa-banner");
      if (staleBanner) staleBanner.remove();
      els.footer.style.display = "block";
      if (!els.messages.querySelector(".gpa-msg")) {
        var hint = el("div", "gpa-msg gpa-bot",
          "Hi! Ask me about GEM-pRF — installation, configuration parameters, running the software, or the paper. " +
          "I answer only from the official GEM-pRF sources, using <strong>" + escapeHtml(chatModel) +
          "</strong> on your machine.");
        els.messages.appendChild(hint);
      }
      warnIfWeakModel();

      setIndexing(true);
      return buildChunkVectors().then(function () {
        // buildParamVectors resolves false rather than rejecting; folding that in keeps the gate airtight.
        return buildParamVectors().then(function (ok) { if (!ok) throw new Error("param embed failed"); });
      }).then(function () {
        setIndexing(false);
      }, function () {
        // A failed build gates the assistant again rather than degrading: retrieval must not run without vectors.
        setIndexing(false);
        var failedModel = embedModel;
        embedModel = null;
        chunkVectors = null;
        paramRows = null;
        connected = false;
        setStatus("off");
        els.footer.style.display = "none";
        var failedBanner = showBanner("Building the retrieval index with <strong>" + escapeHtml(failedModel) +
          "</strong> failed. Make sure it is installed and can serve /api/embed, then retry." +
          "<button class=\"gpa-btn gpa-secondary\" id=\"gpa-retry\">Retry</button>", true);
        var failedRetry = failedBanner.querySelector("#gpa-retry");
        if (failedRetry) failedRetry.addEventListener("click", connect);
      });
    });
  }

  function buildPanel() {
    var panel = el("div");
    panel.id = "gpa-panel";
    panel.innerHTML =
      '<div class="gpa-header">' +
        '<img src="/assets/images/icons/gem-icon-retinotopy-white.png" alt="" onerror="this.style.display=\'none\'">' +
        '<div class="gpa-titlewrap"><div class="gpa-title">GEM-pRF Assistant</div></div>' +
        '<span class="gpa-status-dot" id="gpa-dot" title="connection status"></span>' +
        '<button id="gpa-close" title="Close">×</button>' +
      '</div>' +
      '<div class="gpa-messages" id="gpa-messages"></div>' +
      '<div class="gpa-footer" id="gpa-footer" style="display:none">' +
        '<div class="gpa-modelrow"><span>Model:</span>' +
          '<select id="gpa-model"></select></div>' +
        '<div class="gpa-inputrow">' +
          '<textarea id="gpa-input" rows="1" placeholder="' + INPUT_PLACEHOLDER + '"></textarea>' +
          '<button class="gpa-btn" id="gpa-send">Send</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(panel);

    var launcher = el("button");
    launcher.id = "gpa-launcher";
    launcher.title = "Ask the GEM-pRF Assistant";
    launcher.innerHTML = "&#128172;";
    document.body.appendChild(launcher);

    els.panel = panel;
    els.launcher = launcher;
    els.messages = panel.querySelector("#gpa-messages");
    els.footer = panel.querySelector("#gpa-footer");
    els.input = panel.querySelector("#gpa-input");
    els.send = panel.querySelector("#gpa-send");
    els.dot = panel.querySelector("#gpa-dot");
    els.modelSelect = panel.querySelector("#gpa-model");

    launcher.addEventListener("click", openPanel);
    panel.querySelector("#gpa-close").addEventListener("click", closePanel);
    els.modelSelect.addEventListener("change", function () { chatModel = els.modelSelect.value; warnIfWeakModel(); });
    els.send.addEventListener("click", function () { ask(els.input.value); });
    els.input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(els.input.value); }
    });
    els.input.addEventListener("input", function () {
      els.input.style.height = "auto";
      els.input.style.height = Math.min(els.input.scrollHeight, 120) + "px";
    });
  }

  var bootstrapped = false;
  // Open the panel; load the index and connect on first open.
  function openPanel() {
    els.panel.classList.add("gpa-open");
    els.launcher.classList.add("gpa-hidden");
    if (!bootstrapped) {
      bootstrapped = true;
      ensureIndex().then(connect);
    }
  }
  function closePanel() {
    els.panel.classList.remove("gpa-open");
    els.launcher.classList.remove("gpa-hidden");
  }

  var indexPromise = null;
  // The promise memoizes rejection too: a failed index load stays failed for the page's lifetime.
  function ensureIndex() {
    if (indexPromise) return indexPromise;
    indexPromise = fetch(INDEX_URL, { cache: "no-cache" }).then(function (r) { return r.json(); }).then(function (j) {
      knowledgeIndex = j;
      bm25 = prepareBm25(knowledgeIndex.chunks);
    }).catch(function (e) {
      showBanner("Failed to load the knowledge index: " + (e.message || e), true);
      throw e;
    });
    return indexPromise;
  }

  // Nothing is fetched or probed until the first open.
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", buildPanel);
    } else {
      buildPanel();
    }
  }

  // Node/test surface. These KEY names are the external contract; the functions behind them may be renamed.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      headTerms: headOwners,
      tokenize: tokenize, renderMarkdown: renderMarkdown,
      // Key renamed with the return shape ([spec] -> {named, partial}), so a stale consumer fails loudly.
      matchParametersEmbedding: matchParametersEmbedding, matchParametersByName: matchParametersByName,
      buildParamVectors: buildParamVectors, retrieve: retrieve, buildHumanPrompt: buildHumanPrompt,
      evidenceClass: evidenceClass, hasProseEvidence: hasProseEvidence,
      termFork: forkCandidates, forkQuestion: forkQuestion, forkIsTheSubject: forkIsTheSubject, CLARIFY_DIRECTIVE: CLARIFY_DIRECTIVE,
      resolveForkReply: resolveForkReply, foldClarifyReply: foldClarifyReply, forkReaskQuestion: forkReaskQuestion,
      isCodeRequest: isCodeRequest, codeParamForQuestion: codeParamForQuestion, findCodeChunk: findCodeChunk, findCodeChunks: findCodeChunks,
      namesParameter: namesParameter, valueClarifyingQuestion: valueClarifyFallback,
      needsGoal: needsGoal, engageWithoutEvidence: engageWithoutEvidence,
      INTERACTIVE_NO_EVIDENCE_MESSAGE: INTERACTIVE_NO_EVIDENCE_MESSAGE,
      streamChat: streamChat, SYSTEM_PROMPT: SYSTEM_PROMPT,
      INSUFFICIENT_EVIDENCE_MESSAGE: INSUFFICIENT_EVIDENCE_MESSAGE, isRefusal: isRefusal,
      VALUE_ANSWER_DIRECTIVE: VALUE_ANSWER_DIRECTIVE, GOAL_TRIAD: GOAL_TRIAD,
      contextualize: contextualize, chatComplete: chatComplete,
      isVerificationFollowup: isVerificationFollowup, verifyPriorAnswer: verifyPriorAnswer,
      adjudicateConsistency: adjudicateConsistency, composeVerify: composeVerify, firstSentence: firstSentence,
      analyzeFold: retrieveAndAnswer, generateAnswer: generateAnswer,
      _setHistory: function (h) { history = h; },
      pickEmbedModel: pickEmbedModel, buildChunkVectors: buildChunkVectors, ollamaEmbed: ollamaEmbed,
      // A test swapping the index must clear these three caches.
      _load: function (j) { knowledgeIndex = j; bm25 = prepareBm25(knowledgeIndex.chunks); _tokenOwnersCache = null; _anchorVocabCache = null; _headOwnersCache = null; },
      _setModel: function (m) { chatModel = m; },
      _setEmbedModel: function (m) { embedModel = m; },
      _setModelDigests: function (d) { modelDigests = d; },
      _state: function () { return { embedModel: embedModel, haveVectors: !!chunkVectors, chunks: knowledgeIndex.chunks.length }; }
    };
  }
})();
