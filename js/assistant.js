(function () {
  "use strict";

  var OLLAMA = "http://localhost:11434";
  var DEFAULT_MODEL = "qwen3:14b";
  var PREFERRED_EMBED = "nomic-embed-text";
  var INDEX_URL = "/assistant_index.json";
  var MIN_PARAMS_B = 4.0;
  var TOP_K = 6;
  var HISTORY_TURNS = 10;
  var EVIDENCE_CHAR_CAP = 0;
  var HYBRID_ALPHA = 0.5;
  var ANSWER_CHAR_CAP = 500;
  var LLM_TIMEOUT_MS = 180000;
  var LLM_RETRIES = 3;
  var STREAM_IDLE_MS = 90000;
  var EMBED_PREFIX = {
    "nomic-embed-text": { doc: "search_document: ", query: "search_query: " },
    "mxbai-embed-large": { doc: "", query: "Represent this sentence for searching relevant passages: " },
    "e5": { doc: "passage: ", query: "query: " },
    "bge": { doc: "", query: "Represent this sentence for searching relevant passages: " }
  };

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
    "\"how does it work\"). When an EXPANDED QUERY is provided, treat it as the user's " +
    "intent and use it to interpret which evidence items are relevant.\n" +
    "  - Code is documentation for STRUCTURE. When evidence includes code (function " +
    "bodies, methods, tests), treat the code as authoritative and INFER the structure " +
    "of values, files, and APIs from how the code constructs or consumes them. Example: " +
    "if a serializer writes `json_entry = args2jsonEntry(muX, muY, sigma, r2, ...)`, the " +
    "JSON contains those fields. State the inference plainly and cite the code chunk. " +
    "This license applies to code structure only — not to broader rationales, theory, or " +
    "user-facing semantics.\n" +
    "  - Only respond with 'INSUFFICIENT_EVIDENCE: <one-sentence reason>' when no " +
    "evidence item contains or supports the requested fact at all.\n\n" +
    "Source-type discipline (CRITICAL for refusal correctness):\n" +
    "  source_id prefixes encode evidence type:\n" +
    "    • prose docs: `paper.*`, `website.*` — authoritative for behavior, " +
    "recommendations, UI semantics, defaults, and rationale.\n" +
    "    • XML sample configs: `github.gem.configs.*`, `repo.sample_config`, " +
    "`demokit.sample_configs.*` — illustrative attribute values from example files. " +
    "These are sample data, NOT normative defaults or recommendations.\n" +
    "    • code: `code.*` — authoritative only for code structure and what fields/keys " +
    "exist; not for UI behavior or recommended settings.\n" +
    "  The rule below applies ONLY to three question classes: (a) UI behavior, (b) " +
    "recommended/default values, and (c) runtime semantics of a configurator toggle. For " +
    "these classes ONLY, it OVERRIDES the general 'if ANY evidence contains the fact, " +
    "answer' rule above. For every OTHER question class — definitions, citations, what " +
    "fields/keys exist, code structure, numbers stated in prose — the general rule stands " +
    "and code/XML chunks ARE valid grounding; do not refuse those just because the source " +
    "is code or XML.\n" +
    "  For questions about UI behavior (\"what happens when I check/enable/select X\"), " +
    "recommended or default values (\"what is the default Y\", \"what does GEM-pRF " +
    "recommend\"), or runtime semantics of a configurator toggle, you MUST cite at least " +
    "one prose chunk (`paper.*` or `website.*`) that literally states the behavior. An " +
    "XML attribute value found in a sample config does not establish a default or " +
    "recommendation. A code if/else branch does not establish UI semantics. If only " +
    "XML/code chunks are retrieved and no prose chunk states the behavior, refuse with " +
    "'INSUFFICIENT_EVIDENCE: <reason>' rather than synthesizing the answer from sample " +
    "values or implementation details.\n\n" +
    "Endorsement discipline: A documented default or a value stated in the sources is a FACT " +
    "about what the software does, not an endorsement. Report defaults and stated values plainly " +
    "('the default is X', 'the sample config sets X'). Do NOT assert that a value is 'appropriate', " +
    "'correct', 'the right choice', 'safe to use', 'suitable', or 'recommended' for the user's case " +
    "unless a prose source (`paper.*` or `website.*`) explicitly recommends it, and NEVER manufacture " +
    "certainty ('we can be certain', 'this is definitely appropriate') the evidence does not state.\n\n" +
    "Completeness rule: Include every factual detail the question implicitly asks for. " +
    "Quote numbers, dates, version IDs, software/container names, DOIs, journal volumes, " +
    "and article numbers verbatim from the evidence. If the question asks 'in which " +
    "journal and year', include the journal name AND the year AND the volume AND the " +
    "article identifier when present. If the question asks 'which Docker container', " +
    "include the exact container name.\n\n" +
    "Style rule: Be concise but exact. Prefer short, factual sentences. Write in plain " +
    "English. Do NOT include bracketed source-id citations such as [paper.full] or " +
    "[website.config_generator] in the answer text — provenance is surfaced separately by " +
    "the caller via the retrieved evidence list. Numerical values, defaults, ranges, and " +
    "mathematical formulas in plain notation are fine; quoted strings from log messages are " +
    "fine.\n" +
    "  Code is first-class evidence, exactly like the paper and the docs — not something to " +
    "be hidden. When the question asks about the source, implementation, or the exact " +
    "code/lines/logic, AND a `code.*` evidence item contains it, quote the relevant lines " +
    "verbatim in a fenced code block and then explain in plain English what they do. Quote " +
    "only what a code chunk literally contains; never reconstruct or invent code that is not " +
    "in the evidence.\n" +
    "  For every OTHER question class (definitions, behaviour, recommendations, 'why' " +
    "explanations), answer in plain English and do NOT sprinkle raw identifiers into the " +
    "prose: refer to configurator fields by their UI labels (e.g. 'Normalize HRF', 'Default " +
    "GPU', 'Refine Fitting') rather than their XML paths (e.g. /root/stimulus/binarization/" +
    "@enable), module or function names (e.g. gem.init_setup.manage_gpus, np.linspace), or " +
    "environment variables (e.g. os.environ['CUDA_VISIBLE_DEVICES']). The distinction is " +
    "purpose, not source type: quote code when the user wants the code, describe it in plain " +
    "English when the user wants the concept.\n\n" +
    "Do not invent numbers, citations, authors, container names, version IDs, " +
    "affiliations, rationales, mechanisms, or explanations that are not present in the " +
    "evidence.\n\n" +
    "A CONVERSATION HISTORY block may precede the question. Use it ONLY to resolve what " +
    "the question refers to (e.g. 'it', 'that parameter', a short follow-up). Prior " +
    "answers are NOT evidence: every factual claim must still be grounded in the " +
    "evidence items below, and the refusal rules apply to the resolved question.";

  var INSUFFICIENT_EVIDENCE_MESSAGE =
    "I do not have enough support in the allowed GEM-pRF sources to answer that reliably. " +
    "This prototype is restricted to the paper, GEM-pRF docs, and the published package code.";

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

  var VALUE_ANSWER_DIRECTIVE =
    "\n\nThe user is choosing a value for a setting and has now given their goal. Using ONLY the evidence, " +
    "write a short natural-prose answer (no headings, no numbered list). First, explain the governing " +
    "logic from the evidence -- what this value controls and what a good choice depends on (another " +
    "setting, the stimulus, the dataset, or the hardware) -- reasoning toward the goal they gave. Then, if " +
    "the evidence documents a specific value or default, state it as such; if the evidence documents NO " +
    "specific value for their case, say so plainly. Do NOT invent, estimate, or recall a value, number, or " +
    "formula that is not in the evidence -- not even a commonly-used field convention (e.g. do not say a " +
    "value 'is commonly used'). Do NOT end with a question. Stop after stating what the evidence does and " +
    "does not provide.";

  var INTERACTIVE_NO_EVIDENCE_MESSAGE =
    "I don't have a grounded answer for that in the GEM-pRF paper, docs, or package code. If you're " +
    "configuring a setting, tell me which one and what you're optimizing for -- accuracy, runtime, or " +
    "GPU memory -- and I'll answer from the documentation.";

  var NO_EVIDENCE_TURN_SYSTEM =
    "You are the GEM-pRF configuration assistant. GEM-pRF configures population receptive field (pRF) " +
    "analyses. The user's latest message retrieved NO supporting documentation, so you have no evidence " +
    "in front of you. Reply in one or two sentences:\n" +
    "  - If it is conversational or meta (a greeting, thanks, small talk, or asking what you need from " +
    "them), respond naturally and briefly, then steer them toward their configuration: invite them to " +
    "name the setting they're configuring and what they're optimizing for (accuracy, runtime, or GPU " +
    "memory).\n" +
    "  - If it is a factual GEM-pRF question you have no documentation for, say plainly that the GEM-pRF " +
    "sources (paper, docs, package code) do not cover it. Do not guess.\n" +
    "Hard rule: you have NO evidence, so state NO GEM-pRF fact, default, parameter value, formula, or " +
    "behavior, and never recall specifics from outside the provided sources. When unsure whether " +
    "something is conversational or factual, decline and redirect rather than assert.";

  var NEEDS_GOAL_SYSTEM =
    "You triage questions for a GEM-pRF configuration assistant. Some settings have a best value that " +
    "depends on the user's goal (accuracy, runtime, or GPU memory).\n" +
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
    "recommend one. Using the setting's documented purpose below AND your general understanding of how " +
    "such a setting is chosen in practice, ask ONE short, natural clarifying question for the CONCRETE, " +
    "real-world quantity the user actually knows about their own experiment that would determine a good " +
    "value -- for example the scan/run length (or the TR and number of volumes), the dataset size, the " +
    "available GPU memory, or the stimulus's size on screen. Prefer that operational input over an " +
    "abstract internal quantity: e.g. for a low-frequency-drift regressor, ask how long their runs are " +
    "and what high-pass cutoff they want, NOT 'how much drift do you expect'; for a memory or batch " +
    "setting, ask their dataset size and GPU memory; for a stimulus setting, ask the stimulus's extent on " +
    "screen. You may use general knowledge to decide WHICH input to ask for, but you must NOT assert any " +
    "GEM-pRF fact, default, formula, or value -- only ask the question. If more than one distinct setting " +
    "matches the user's term, ask which they mean as part of the same question. You may add that they can " +
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
    "NOT graft that prior context onto it. Resolve references from what the user previously ASKED; do " +
    "not import a specific parameter or setting name from the assistant's ANSWER unless the latest " +
    "question explicitly refers to it (a new topic in the latest question overrides a prior answer's " +
    "subject). Do not answer it, do not add new topics -- output only the rewritten question.";

  var STOPWORDS = new Set(("a an the of to in on for and or is are was were be been being do does did " +
    "how what which when where why who whom this that these those it its as at by with from into " +
    "can could should would will i you we they he she them his her their our your me my mine " +
    "if then than so such not no yes about over under between out up down off also just only " +
    "gem prf gemprf does use used using get set").split(/\s+/));

  var idx = null;
  var bm25 = null;
  var models = [];
  var embedModels = [];
  var chatModel = DEFAULT_MODEL;
  var embedModel = null;
  var chunkVectors = null;
  var paramRows = null;
  var connected = false;
  var busy = false;
  var history = [];
  var activeBot = null;
  var pendingClarify = null;
  var valueChoiceActive = false;
  var els = {};


  // Lowercase word tokens, minus stopwords and short fragments.
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


  // Build the BM25 model over the corpus chunks.
  function prepareBm25(chunks) {
    var docs = chunks.map(function (c) {
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

  function bm25Score(model, weightedTerms) {
    var k1 = 1.5, b = 0.75;
    var scores = new Array(model.N).fill(0);
    for (var d = 0; d < model.N; d++) {
      var s = 0;
      var dl = model.len[d];
      for (var term in weightedTerms) {
        var f = model.tf[d][term];
        if (!f) continue;
        var n = model.df[term] || 0;
        var idfv = Math.log(1 + (model.N - n + 0.5) / (n + 0.5));
        var denom = f + k1 * (1 - b + b * dl / model.avgdl);
        s += weightedTerms[term] * idfv * (f * (k1 + 1)) / denom;
      }
      scores[d] = s;
    }
    return scores;
  }

  var MIN_PARAMETER_SCORE = 0.32;
  var PARAMETER_CUTOFF_MARGIN = 0.05;
  var MAX_RETURNED_PARAMETERS = 4;

  function paramRowTexts() {
    var rows = [];
    (idx.parameters || []).forEach(function (p) {
      rows.push({ pid: p.id, text: p.label + ". Aliases: " + ((p.aliases || []).join(", ")) + ". Identifier: " + p.id + "." });
      rows.push({ pid: p.id, text: "XML path: " + (p.xml_path || "") + ". Summary: " + (p.summary || "") + " Significance: " + (p.significance || "") });
    });
    return rows;
  }

  // Match parameters by cosine over the catalog rows (two-stage gate).
  function matchParametersEmbedding(qVec) {
    var best = Object.create(null);
    for (var i = 0; i < paramRows.length; i++) {
      var s = dot(qVec, paramRows[i].vec), pid = paramRows[i].pid;
      if (best[pid] === undefined || s > best[pid]) best[pid] = s;
    }
    var ordered = Object.keys(best).map(function (pid) { return [pid, best[pid]]; })
      .sort(function (a, b) { return b[1] - a[1]; });
    if (!ordered.length) return { specs: [], scores: {} };
    var cutoff = Math.max(MIN_PARAMETER_SCORE, ordered[0][1] - PARAMETER_CUTOFF_MARGIN);
    var lookup = Object.create(null);
    (idx.parameters || []).forEach(function (p) { lookup[p.id] = p; });
    var specs = [], scores = {};
    for (var k = 0; k < ordered.length && specs.length < MAX_RETURNED_PARAMETERS; k++) {
      if (ordered[k][1] < cutoff) break;
      if (lookup[ordered[k][0]]) { specs.push(lookup[ordered[k][0]]); scores[ordered[k][0]] = ordered[k][1]; }
    }
    return { specs: specs, scores: scores };
  }

  // Match parameters by label/alias substring (BM25-only fallback).
  function matchParametersSubstring(question) {
    var ql = " " + question.toLowerCase() + " ";
    var matched = [];
    (idx.parameters || []).forEach(function (p) {
      var probes = [p.label].concat(p.aliases || []);
      var hits = 0, best = 0;
      for (var i = 0; i < probes.length; i++) {
        var probe = (probes[i] || "").toLowerCase().trim();
        if (probe.length >= 3 && ql.indexOf(probe) !== -1) { hits++; best = Math.max(best, probe.length); }
      }
      if (hits > 0) matched.push({ spec: p, score: hits * 1000 + best });
    });
    matched.sort(function (a, b) { return b.score - a.score; });
    return matched.slice(0, MAX_RETURNED_PARAMETERS).map(function (m) { return m.spec; });
  }

  var _tokenOwnersCache = null;
  var _anchorVocabCache = null;

  // Content tokens (>2 chars) used for term disambiguation.
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
    (idx.parameters || []).forEach(function (p) {
      disambigTokens([p.label].concat(p.aliases || []).join(" ")).forEach(function (t) {
        (owners[t] = owners[t] || Object.create(null))[p.id] = 1;
      });
    });
    _tokenOwnersCache = owners;
    return owners;
  }

  // Fork the parameters an ambiguous shared term leaves open (two-phase).
  function termFork(question, matchedSpecs) {
    var owners = tokenOwners();
    var owned = disambigTokens(question).filter(function (t) { return owners[t]; }).map(function (t) { return owners[t]; });
    var ambiguous = owned.filter(function (o) { return Object.keys(o).length >= 2; });
    if (!ambiguous.length) return [];
    var candSet = Object.create(null);
    Object.keys(ambiguous[0]).forEach(function (pid) {
      if (ambiguous.every(function (o) { return o[pid]; })) candSet[pid] = 1;
    });
    if (Object.keys(candSet).length < 2) return [];
    owned.forEach(function (o) {
      var keys = Object.keys(candSet);
      var overlap = keys.filter(function (pid) { return o[pid]; });
      if (overlap.length > 0 && overlap.length < keys.length) {
        candSet = Object.create(null);
        overlap.forEach(function (pid) { candSet[pid] = 1; });
      }
    });
    var hits = (matchedSpecs || []).filter(function (spec) { return candSet[spec.id]; });
    return hits.length >= 2 ? hits.slice(0, 4) : [];
  }

  // Name each forked parameter so the user can pick one.
  function forkQuestion(fork) {
    var labels = fork.map(function (spec) { return spec.label; });
    var joined = labels.length === 2 ? labels[0] + " or " + labels[1]
      : labels.slice(0, -1).join(", ") + ", or " + labels[labels.length - 1];
    return "GEM-pRF has more than one setting that matches that -- which do you mean: " + joined + "?";
  }

  // Terms that identify a specific parameter.
  function anchorVocab() {
    if (_anchorVocabCache) return _anchorVocabCache;
    var tokens = Object.create(null), phrases = [], collapsed = [];
    (idx.parameters || []).forEach(function (p) {
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
    var v = anchorVocab();
    var low = String(question || "").toLowerCase();
    if (disambigTokens(question).some(function (t) { return v.tokens[t]; })) return true;
    if (v.phrases.some(function (ph) { return low.indexOf(ph) !== -1; })) return true;
    var compact = low.replace(/[^a-z0-9]/g, "");
    return v.collapsed.some(function (c) { return compact.indexOf(c) !== -1; });
  }

  // Fixed value-choice clarify asking the user's goal (fallback template).
  function valueClarifyingQuestion(fork, namesSetting) {
    var goal = "what are you optimizing for -- accuracy, runtime, or GPU memory?";
    if (fork.length) {
      var labels = fork.map(function (spec) { return spec.label; });
      var which = labels.length === 2 ? labels[0] + " or " + labels[1]
        : labels.slice(0, -1).join(", ") + ", or " + labels[labels.length - 1];
      return "GEM-pRF has more than one setting that matches that -- which do you mean: " + which + "? And " + goal;
    }
    if (namesSetting) return "That depends on your goal -- " + goal + " (Or say 'just the default' for the documented value.)";
    return "Which setting are you choosing a value for, and " + goal + " (Or name the setting and say 'just the default'.)";
  }

  // Generate a context-tailored value-choice clarifying question.
  function generateValueClarify(question, ret, fork) {
    var focus = fork.length ? fork : (ret.specs || []).slice(0, 1);
    var evidence = (ret.chunks || []).slice(0, 3)
      .map(function (c) { return "- " + (c.heading || c.source_id) + ": " + c.text.split(/\s+/).join(" ").slice(0, 260); })
      .join("\n") || "- none";
    var human = "User question: " + question + "\n\nThe setting the user is asking about:\n" +
      parameterContext(focus) + "\n\nWhat the documentation says:\n" + evidence +
      (fork.length ? "\n\nDistinct settings that match the user's term (ask which one they mean): " +
        fork.map(function (s) { return s.label; }).join(", ") : "") +
      "\n\nYour one clarifying question:";
    return chatComplete([
      { role: "system", content: VALUE_CLARIFY_SYSTEM },
      { role: "user", content: human }
    ]).then(function (t) {
      t = (t || "").replace(/^["'`]+|["'`]+$/g, "").trim();
      return t || valueClarifyingQuestion(fork, namesParameter(question));
    }).catch(function () { return valueClarifyingQuestion(fork, namesParameter(question)); });
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

  function dot(a, b) {
    var s = 0, n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }

  // int8-quantized: full-float JSON overflows the ~5MB localStorage quota; the ~1/127 error is negligible for cosine ranking.
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
    var u8 = new Uint8Array(i8.buffer), CH = 0x8000, s = "";
    for (var k = 0; k < u8.length; k += CH) s += String.fromCharCode.apply(null, u8.subarray(k, k + CH));
    return dim + ":" + btoa(s);
  }

  // Decode the int8 vector cache back to unit Float32 vectors.
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

  // Embed texts via the visitor's Ollama (batch, legacy fallback).
  function ollamaEmbed(model, texts, kind) {
    var pref = embedPrefix(model, kind);
    var inputs = texts.map(function (t) { return pref + t; });
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

  // Pick the embedding model to use, if any is installed.
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

  // Embed the corpus chunks once, cached by (corpus_sha, model).
  function buildChunkVectors(onCompute) {
    var cacheKey = "gpa:vec2:" + idx.meta.corpus_sha + ":" + embedModel;
    try {
      var decoded = decodeVectors(localStorage.getItem(cacheKey) || "", idx.chunks.length);
      if (decoded) { chunkVectors = decoded; return Promise.resolve(true); }
    } catch (e) {  }

    if (onCompute) onCompute();
    var texts = idx.chunks.map(function (c) { return c.heading + "\n" + c.text; });
    return ollamaEmbed(embedModel, texts, "doc").then(function (vecs) {
      chunkVectors = vecs;
      try { localStorage.setItem(cacheKey, encodeVectors(vecs)); } catch (e) {  }
      return true;
    });
  }

  // Embed the parameter-catalog rows once (cached like the chunks).
  function buildParamVectors() {
    var rows = paramRowTexts();
    if (!rows.length) { paramRows = []; return Promise.resolve(true); }
    var cacheKey = "gpa:pvec2:" + idx.meta.corpus_sha + ":" + embedModel;
    try {
      var decoded = decodeVectors(localStorage.getItem(cacheKey) || "", rows.length);
      if (decoded) { for (var i = 0; i < rows.length; i++) rows[i].vec = decoded[i]; paramRows = rows; return Promise.resolve(true); }
    } catch (e) {  }
    return ollamaEmbed(embedModel, rows.map(function (r) { return r.text; }), "doc").then(function (vecs) {
      for (var j = 0; j < rows.length; j++) rows[j].vec = vecs[j];
      paramRows = rows;
      try { localStorage.setItem(cacheKey, encodeVectors(vecs)); } catch (e) {  }
      return true;
    }).catch(function () { paramRows = null; return false; });
  }


  function minmax(arr) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < arr.length; i++) { if (arr[i] < lo) lo = arr[i]; if (arr[i] > hi) hi = arr[i]; }
    var rng = hi - lo, out = new Array(arr.length);
    for (var j = 0; j < arr.length; j++) out[j] = rng > 0 ? (arr[j] - lo) / rng : 0;
    return out;
  }

  // Fuse semantic and BM25 scores (min-max, alpha blend).
  function fuse(bm25Scores, semScores) {
    if (!semScores) return bm25Scores;
    var nb = minmax(bm25Scores), ns = minmax(semScores);
    var out = new Array(bm25Scores.length);
    for (var i = 0; i < out.length; i++) out[i] = HYBRID_ALPHA * ns[i] + (1 - HYBRID_ALPHA) * nb[i];
    return out;
  }

  // Retrieve evidence + matched params: hybrid, or BM25-only without embeddings.
  function retrieve(question) {
    var haveEmbed = embedModel && chunkVectors && paramRows;
    var qPromise = haveEmbed
      ? ollamaEmbed(embedModel, [question], "query").then(function (qv) { return qv[0]; }).catch(function () { return null; })
      : Promise.resolve(null);

    return qPromise.then(function (qVec) {
      var matched;
      if (qVec) {
        matched = matchParametersEmbedding(qVec);
        // Literal name = confident match (1.0): nomic embeds rare acronyms (e.g. nDCT) poorly.
        matchParametersSubstring(question).forEach(function (spec) {
          if (matched.scores[spec.id] === undefined) matched.specs.push(spec);
          matched.scores[spec.id] = 1.0;
        });
        matched.specs.sort(function (a, b) { return (matched.scores[b.id] || 0) - (matched.scores[a.id] || 0); });
        matched.specs = matched.specs.slice(0, MAX_RETURNED_PARAMETERS);
      } else {
        matched = { specs: matchParametersSubstring(question), scores: {} };
        matched.specs.forEach(function (s) { matched.scores[s.id] = 1.0; });
      }
      var matchedSpecs = matched.specs;

      var weights = Object.create(null);
      tokenize(question).forEach(function (t) { weights[t] = Math.max(weights[t] || 0, 1); });
      var expansionWords = [];
      matchedSpecs.forEach(function (p) {
        tokenize([p.label, (p.aliases || []).join(" "), p.summary].join(" ")).forEach(function (t) {
          if (!(t in weights)) expansionWords.push(t);
          weights[t] = Math.max(weights[t] || 0, 0.6);
        });
      });
      var bm25Scores = bm25Score(bm25, weights);
      var semScores = qVec ? chunkVectors.map(function (cv) { return dot(qVec, cv); }) : null;

      var fused = fuse(bm25Scores, semScores);
      var ranked = [];
      for (var i = 0; i < fused.length; i++) {
        if (fused[i] > 0) ranked.push({ i: i, score: fused[i] });
      }
      ranked.sort(function (a, b) { return b.score - a.score; });

      var perSource = Object.create(null);
      var picked = [];
      for (var r = 0; r < ranked.length && picked.length < TOP_K; r++) {
        var c = idx.chunks[ranked[r].i];
        var sid = c.source_id || "?";
        if ((perSource[sid] || 0) >= 3) continue;
        perSource[sid] = (perSource[sid] || 0) + 1;
        picked.push(c);
      }
      return { chunks: picked, specs: matchedSpecs, paramScores: matched.scores, expansion: expansionWords,
               mode: semScores ? "hybrid" : "bm25" };
    });
  }


  function capText(text, cap) {
    if (cap <= 0 || text.length <= cap) return text;
    return text.slice(0, cap).replace(/\s+\S*$/, "") + " …";
  }

  function evidenceContext(chunks) {
    if (!chunks.length) return "- none";
    return chunks.map(function (c) {
      return "[" + c.source_id + " " + c.heading + "] " + capText(c.text, EVIDENCE_CHAR_CAP);
    }).join("\n\n");
  }

  function parameterContext(specs) {
    if (!specs.length) return "- none";
    return specs.map(function (p) {
      return "- " + p.label + " (" + p.xml_path + "): " + p.summary + " " + p.significance;
    }).join("\n");
  }

  function renderTurns() {
    return history.slice(-HISTORY_TURNS).map(function (t) {
      return "User: " + t.q + "\nAssistant: " + t.a;
    }).join("\n");
  }

  function historyContext() { return history.length ? renderTurns() : "- none"; }

  // Record an answered turn in the rolling history (capped).
  function recordTurn(q, a) {
    a = (a || "").split(/\s+/).join(" ").trim();
    if (a.length > ANSWER_CHAR_CAP) a = a.slice(0, ANSWER_CHAR_CAP).replace(/\s+\S*$/, "") + " …";
    history.push({ q: (q || "").trim(), a: a });
    if (history.length > HISTORY_TURNS) history = history.slice(-HISTORY_TURNS);
  }

  // Build the human prompt; history feeds it only for genuine follow-ups.
  function buildHumanPrompt(question, ret, isFollowup, allowClarify, valueAnswer) {
    return "Conversation history (reference resolution only; NOT evidence):\n" +
      (isFollowup ? historyContext() : "- none") + "\n\n" +
      "Question: " + question + "\n\n" +
      "Expanded query (intent inference; cite from the original question only):\n" +
      "- none" + "\n\n" +
      "Matched parameters:\n" + parameterContext(ret.specs) + "\n\n" +
      "Evidence (each item is `[source_id heading_path] text`):\n" + evidenceContext(ret.chunks) +
      (allowClarify ? CLARIFY_DIRECTIVE : "") + (valueAnswer ? VALUE_ANSWER_DIRECTIVE : "");
  }


  function withTimeout(ms) {
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

  // Model size in billions from /api/show, or null.
  function modelParamsB(model) {
    return fetch(OLLAMA + "/api/show", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model }),
      signal: withTimeout(5000).signal
    }).then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) { return j ? parseParamsB(j.details && j.details.parameter_size) : null; })
      .catch(function () { return null; });
  }

  // Probe Ollama for available chat and embedding models.
  function probeOllama() {
    return fetch(OLLAMA + "/api/tags", { signal: withTimeout(3500).signal })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (j) {
        var all = (j.models || []).map(function (m) { return m.name; });
        var isEmbed = function (n) { return /embed|bge|minilm|nomic|mxbai|arctic|gte|\be5\b/i.test(n); };
        models = all.filter(function (n) { return !isEmbed(n); });
        embedModels = all.filter(isEmbed);
        return { ok: true, models: models, embedModels: embedModels };
      })
      .catch(function (e) { return { ok: false, error: e.message || String(e) }; });
  }

  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  // Non-streaming chat completion with timeout and retry.
  function chatComplete(messages, attempt) {
    attempt = attempt || 0;
    return fetch(OLLAMA + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: chatModel, messages: messages, stream: false, think: false, options: { temperature: 0 } }),
      signal: withTimeout(LLM_TIMEOUT_MS).signal
    }).then(function (r) {
      if (!r.ok) throw new Error("Ollama HTTP " + r.status);
      return r.json();
    }).then(function (j) { return (j.message && j.message.content ? j.message.content : "").trim(); })
      .catch(function (e) {
        if (attempt < LLM_RETRIES - 1) return delay(1500 * (attempt + 1)).then(function () { return chatComplete(messages, attempt + 1); });
        throw e;
      });
  }

  function normQ(s) { return (s || "").toLowerCase().split(/\s+/).join(" ").replace(/^[\s?.!]+|[\s?.!]+$/g, ""); }

  // Fold a follow-up's referents into a standalone question before retrieval.
  function contextualize(question) {
    if (!history.length) return Promise.resolve(question);
    var hist = renderTurns();
    var messages = [
      { role: "system", content: CONDENSE_SYSTEM },
      { role: "user", content: "Conversation:\n" + hist + "\n\nLatest question: " + question + "\n\nStandalone question:" }
    ];
    return chatComplete(messages).then(function (text) {
      return (text || "").replace(/^["'`]+|["'`]+$/g, "").trim() || question;
    }).catch(function () { return question; });
  }

  // Classify whether a value-choice question needs the user's goal first.
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

  // Stream the grounded answer token by token from the LLM.
  function streamChat(question, ret, onToken, isFollowup, allowClarify, valueAnswer) {
    var body = {
      model: chatModel,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: buildHumanPrompt(question, ret, isFollowup, allowClarify, valueAnswer) }
      ],
      stream: true,
      think: false,  // = Python reasoning_effort=none
      options: { temperature: 0 }
    };
    // Idle timeout re-armed per chunk: kill a stalled stream, never a slow-but-advancing CPU one.
    var ctl = new AbortController();
    var idle;
    function arm() { idle = setTimeout(function () { ctl.abort(); }, STREAM_IDLE_MS); }
    function disarm() { clearTimeout(idle); }
    arm();
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
              onToken(full);
            }
          }
          arm();
          return pump();
        });
      }
      return pump();
    });
  }

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

  function normalizeTurn(text) {
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
    var norm = normalizeTurn(question);
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

  // Non-streaming answer on a query; returns {answer, refused}.
  function generateAnswer(question, ret) {
    return chatComplete([
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: buildHumanPrompt(question, ret) }
    ]).then(function (raw) {
      if (/INSUFFICIENT_EVIDENCE/i.test(raw)) return { answer: raw, refused: true };
      return { answer: raw, refused: false };
    });
  }

  // Retrieve + generate on a folded query and report whether it's grounded.
  function analyzeFold(query) {
    return retrieve(query).then(function (ret) {
      return generateAnswer(query, ret).then(function (g) {
        return { ret: ret, answer: g.answer, answered: !g.refused };
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

  // Adjudicate the prior answer against fresh evidence (verify framing).
  function adjudicateConsistency(prior, fresh) {
    return chatComplete([
      { role: "system", content: ADJUDICATE_SYSTEM },
      { role: "user", content: "EARLIER answer:\n" + prior.slice(0, 1500) + "\n\nFRESH answer:\n" + fresh.slice(0, 1500) + "\n\nOne word:" }
    ]).then(function (t) { var tok = firstToken(t); return ADJUDICATE_VERDICTS[tok] ? tok : ""; })
      .catch(function () { return ""; });
  }

  // Compose the verify confirm-or-correct message.
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
    var seed = (prior.q + " " + firstSentence(prior.a)).trim();
    return analyzeFold(seed).then(function (r) {
      if (!r.answered) {
        fillBubble("I can't re-confirm that from the allowed GEM-pRF sources right now — " +
          "re-checking, I don't find support for it.");
        return true;
      }
      return adjudicateConsistency(prior.a, r.answer).then(function (verdict) {
        fillBubble(composeVerify(verdict, r.answer));
        renderCitations(activeBot, r.ret.chunks);
        return true;
      });
    });
  }

  function fillBubble(html) {
    activeBot.classList.remove("gpa-cursor");
    activeBot.innerHTML = renderMarkdown(html);
    scrollDown();
  }


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

  // Render the source citations under an answer.
  function renderCitations(node, chunks) {
    if (!chunks.length) return;
    var seen = Object.create(null);
    var items = [];
    chunks.forEach(function (c) {
      var src = (idx.sources && idx.sources[c.source_id]) || {};
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

  // Set the pending single-reply clarify state and show the question.
  function askClarify(question, clarifyingQuestion, isValueChoice) {
    pendingClarify = { question: question, valueChoice: !!isValueChoice };
    fillBubble(clarifyingQuestion);
  }

  var CODE_EXPLAIN_SYSTEM =
    "The user asked to see GEM-pRF source code, and the exact code (one or more snippets) has ALREADY been " +
    "shown to them above. In 1-4 plain-English sentences, explain what it does; when several snippets are " +
    "shown, say how they relate (e.g. where a value is parsed, passed, and used). Do NOT reproduce, rewrite, " +
    "quote, or include any code, identifiers-as-code, or pseudocode -- it is already displayed. Describe only " +
    "behaviour and purpose.";

  // True when the user asks to see source code.
  function isCodeRequest(q) {
    return /\bcode\b|\bimplementation\b|\bsource[- ]?code\b/i.test(q || "");
  }

  // Find the parameter the question names (id/label/alias match).
  function codeParamForQuestion(question) {
    var q = " " + String(question || "").toLowerCase() + " ";
    var qSpaces = q.replace(/[_.]/g, " ");
    var best = null, bestLen = 0;
    (idx.parameters || []).forEach(function (p) {
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

  // Trim a size-split chunk to the lines where the parameter appears (plus a little context). Prefers the
  // quoted config-key line (the real usage) over bare mentions, so a config read isn't buried in machinery.
  function focusExcerpt(text, terms) {
    var distinctive = terms.filter(function (w) { return w.length >= 3 && !GENERIC_CODE[w]; });
    if (!distinctive.length) return text;
    var lines = text.split("\n");
    function hitsFor(preds) {
      var h = [];
      lines.forEach(function (ln, i) { var low = ln.toLowerCase(); for (var k = 0; k < preds.length; k++) { if (low.indexOf(preds[k]) !== -1) { h.push(i); break; } } });
      return h;
    }
    var quoted = [];
    distinctive.forEach(function (w) { quoted.push('"' + w + '"'); quoted.push("'" + w + "'"); });
    var hit = hitsFor(quoted);            // the config-key line, if any
    if (!hit.length) hit = hitsFor(distinctive);  // else any mention
    if (!hit.length) return text;
    var CTX = 4;
    return lines.slice(Math.max(0, hit[0] - CTX), Math.min(lines.length, hit[hit.length - 1] + CTX + 1)).join("\n");
  }

  // True when a chunk reads the parameter as a quoted config key -- then even the primary block is trimmed.
  function isConfigReadChunk(text, terms) {
    var low = String(text || "").toLowerCase();
    var d = terms.filter(function (w) { return w.length >= 3 && !GENERIC_CODE[w]; });
    for (var k = 0; k < d.length; k++) { if (low.indexOf('"' + d[k] + '"') !== -1 || low.indexOf("'" + d[k] + "'") !== -1) return true; }
    return false;
  }

  // Every curated code site for the named parameter: the best chunk per source file in its code_source_ids,
  // most param-dense first. Falls back to the single best retrieved code chunk when nothing is clearly named.
  function findCodeChunks(question, ret) {
    var specs = ret.specs || [];
    var primary = codeParamForQuestion(question) || specs[0];
    var isCode = function (c) {
      var s = idx.sources[c.source_id];
      return s && s.kind === "code" && !/tests?[._/]|_test|test_/i.test(c.source_id);
    };
    var terms = primary ? tokenize([primary.label].concat(primary.aliases || []).join(" ")) : tokenize(question);
    function score(c) {
      var t = c.text.toLowerCase(), s = 0;
      terms.forEach(function (w) { var i = 0, n = 0; while ((i = t.indexOf(w, i)) !== -1) { n++; i += w.length; } s += n; });
      if (/(^|\n)\s*(def|class)\s/.test(c.text)) s += 0.5;
      return s;
    }
    var distinctive = terms.filter(function (w) { return w.length >= 3 && !GENERIC_CODE[w]; });
    // The parameter as a quoted config key (e.g. cfg.measured_data["batches"]) marks the real usage site,
    // which a bare-word count misses when the term also names internal variables.
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
      chunks.forEach(function (c) { var base = score(c), b = base + configBoost(c); if (b > bestBoosted) { best = c; bestBoosted = b; bestBase = base; } });
      return { chunk: best, base: bestBase, boosted: bestBoosted };
    }
    var sids = primary ? (primary.code_source_ids || []) : [];
    var picked = [];
    sids.forEach(function (sid) {
      var chunks = idx.chunks.filter(function (c) { return c.source_id === sid && isCode(c); });
      if (!chunks.length) return;
      var b = bestIn(chunks);
      if (b.chunk && b.boosted > 0) picked.push(b);  // only sources where the parameter actually appears
    });
    if (picked.length) {
      picked.sort(function (a, b) { return b.base - a.base; });  // implementation (most param-dense) leads
      return picked.slice(0, 5).map(function (p) { return p.chunk; });
    }
    var pool = ret.chunks.filter(isCode);  // no curated match: best retrieved code chunk, or nothing
    var f = bestIn(pool);
    return f.chunk ? [f.chunk] : [];
  }

  // Single top code chunk (the named parameter's curated code); thin wrapper over findCodeChunks.
  function findCodeChunk(question, ret) {
    var arr = findCodeChunks(question, ret);
    return arr.length ? arr[0] : null;
  }

  // Show code verbatim: the model rewrites real code (wrong signatures, dropped lines) when asked to quote it.
  // Renders every curated code site for the parameter, each labelled by its source, then one explanation.
  function showCodeVerbatim(question, ret, codeChunks) {
    var primary = codeParamForQuestion(question) || (ret.specs && ret.specs[0]);
    var terms = primary ? tokenize([primary.label].concat(primary.aliases || []).join(" ")) : tokenize(question);
    var blocks = codeChunks.map(function (c, i) {
      var raw = String(c.text || "").replace(/^[^\n]{1,160} > [^\n]{0,160}\n+/, "").trim();
      // Full only for a primary that's an actual implementation; a config-read or secondary site is trimmed.
      var code = (i === 0 && !isConfigReadChunk(raw, terms)) ? raw : focusExcerpt(raw, terms);
      var title = (idx.sources[c.source_id] && idx.sources[c.source_id].title) || c.source_id;
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
      renderCitations(activeBot, codeChunks.concat(ret.chunks));
      recordTurn(question, "Showed source code from " + srcList + ". " + (expl || "").trim());
      scrollDown();
    }).catch(function () {
      activeBot.innerHTML = renderMarkdown(md);
      renderCitations(activeBot, codeChunks.concat(ret.chunks));
      recordTurn(question, "Showed source code from " + srcList + ".");
      scrollDown();
    });
  }

  // Normal answer path: stream, refuse honestly, or ask one CLARIFY question.
  function answerNormally(question, ret, resolved, allowClarify, valueAnswer) {
    var refusalMsg = allowClarify ? INTERACTIVE_NO_EVIDENCE_MESSAGE : INSUFFICIENT_EVIDENCE_MESSAGE;
    if (!ret.chunks.length) {
      var noEv = allowClarify ? engageWithoutEvidence(question) : Promise.resolve(INSUFFICIENT_EVIDENCE_MESSAGE);
      return noEv.then(function (text) { fillBubble(text); recordTurn(question, text); });
    }
    if (isCodeRequest(resolved || question)) {
      var codeChunks = findCodeChunks(resolved || question, ret);
      if (codeChunks.length) return showCodeVerbatim(question, ret, codeChunks);
    }
    var isFollowup = !!resolved && normQ(resolved) !== normQ(question);
    var rendered = "";
    function isClarify(t) { return allowClarify && /^\s*CLARIFY:/i.test((t || "").trim()); }
    return streamChat(question, ret, function (full) {
      rendered = full;
      if (isClarify(full)) activeBot.innerHTML = renderMarkdown(full.trim().replace(/^\s*CLARIFY:\s*/i, ""));
      else if (/INSUFFICIENT_EVIDENCE/i.test(full)) activeBot.innerHTML = renderMarkdown(refusalMsg);
      else activeBot.innerHTML = renderMarkdown(full);
      scrollDown();
    }, isFollowup, allowClarify, valueAnswer).then(function () {
      activeBot.classList.remove("gpa-cursor");
      if (isClarify(rendered)) {
        var q = rendered.trim().replace(/^\s*CLARIFY:\s*/i, "").trim() || "Could you say a bit more about what you're trying to do?";
        askClarify(question, q);
        return;
      }
      var refused = /INSUFFICIENT_EVIDENCE/i.test(rendered);
      var answer = refused ? refusalMsg : rendered.trim();
      activeBot.innerHTML = renderMarkdown(answer);
      if (!refused) renderCitations(activeBot, ret.chunks);
      recordTurn(question, answer);
      scrollDown();
    });
  }

  // Core turn: retrieve, then answer, clarify, or show code.
  function corePipeline(question, allowClarify, valueAnswer) {
    return contextualize(question).then(function (contextual) {
      var standalone = (normQ(question) !== normQ(contextual)) ? contextual : question;
      var isFollowup = normQ(standalone) !== normQ(question);
      if (allowClarify && valueChoiceActive && !isFollowup) valueChoiceActive = false;
      var continuing = allowClarify && valueChoiceActive && isFollowup;
      var goalP = (allowClarify && !continuing) ? needsGoal(contextual) : Promise.resolve(false);
      return Promise.all([retrieve(contextual), goalP]).then(function (r) {
        var ret = r[0], goal = r[1];
        if (allowClarify && !continuing) {
          var fork = termFork(contextual, ret.specs);
          if (goal) { valueChoiceActive = true; return generateValueClarify(contextual, ret, fork).then(function (q) { askClarify(question, q, true); }); }
          if (fork.length) { askClarify(question, forkQuestion(fork), false); return; }
        }
        return answerNormally(question, ret, standalone, allowClarify && !continuing, valueAnswer);
      });
    });
  }

  // Route a question through condense, verify, and value-choice before answering.
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
    if (busy || !connected) return;
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

    if (pendingClarify) {
      var original = pendingClarify.question;
      var valueChoice = pendingClarify.valueChoice;
      pendingClarify = null;
      var keep = question && !/^(quit|exit|cancel|stop)$/i.test(question);
      var folded = keep ? original + " (" + question + ")" : original;
      corePipeline(folded, false, valueChoice).catch(fail).then(done);
      return;
    }

    processQuestion(question).catch(fail).then(done);
  }

  // Show a status/setup banner in the message area.
  function showBanner(html, isError) {
    var existing = els.messages.querySelector(".gpa-banner");
    if (existing) existing.remove();
    var b = el("div", "gpa-banner" + (isError ? " gpa-error" : ""), html);
    els.messages.appendChild(b);
    scrollDown();
    return b;
  }

  // One-time Ollama setup instructions banner.
  function setupInstructionsHtml() {
    var origin = window.location.origin;
    return "<strong>Connect your local model</strong><br>" +
      "This assistant runs the LLM on <em>your</em> machine — nothing is sent to any server. " +
      "One-time setup:" +
      "<br>1. Install <a href=\"https://ollama.com/download\" target=\"_blank\" rel=\"noopener\">Ollama</a> " +
      "and pull the model:" +
      "<code>ollama pull " + DEFAULT_MODEL + "</code>" +
      "2. (Optional, better retrieval) pull an embedding model for semantic search:" +
      "<code>ollama pull " + PREFERRED_EMBED + "</code>" +
      "3. Allow this site to reach Ollama, then restart it:" +
      "<code>launchctl setenv OLLAMA_ORIGINS \"" + origin + "\"   # macOS\n" +
      "# Linux/manual:  OLLAMA_ORIGINS=\"" + origin + "\" ollama serve</code>" +
      "<button class=\"gpa-btn gpa-secondary\" id=\"gpa-retry\">Retry connection</button>" +
      "<div class=\"gpa-disclaimer\">Note: needs a Chromium-based browser (Chrome/Edge/Brave). Safari blocks localhost requests from HTTPS pages.</div>";
  }

  // Fill the model dropdown from the installed Ollama models.
  function populateModelSelect() {
    els.modelSelect.innerHTML = "";
    models.forEach(function (n) {
      var o = el("option");
      o.value = n; o.textContent = n;
      els.modelSelect.appendChild(o);
    });
    chatModel = models.indexOf(DEFAULT_MODEL) !== -1 ? DEFAULT_MODEL : models[0];
    els.modelSelect.value = chatModel;
  }

  function removeWeakBanner() {
    var w = els.messages.querySelector(".gpa-weakwarn");
    if (w) w.remove();
  }

  function warnIfWeakModel() {
    removeWeakBanner();
    if (!connected) return;
    var model = chatModel;
    modelParamsB(model).then(function (b) {
      if (b === null || b >= MIN_PARAMS_B || chatModel !== model) return;
      var w = el("div", "gpa-banner gpa-weakwarn",
        "<strong>" + escapeHtml(model) + "</strong> has only " + b.toFixed(1) + "B parameters — likely not " +
        "strong enough for reliable grounded answers. Prefer a " + MIN_PARAMS_B.toFixed(0) + "B+ model (e.g. " + DEFAULT_MODEL + ").");
      els.messages.appendChild(w);
      scrollDown();
    });
  }

  // Connect to Ollama: probe, greet, size-gate, then build vectors.
  function connect() {
    var b = showBanner("Connecting to your local Ollama…", false);
    return probeOllama().then(function (res) {
      if (!res.ok) {
        connected = false;
        setStatus("off");
        var banner = showBanner(setupInstructionsHtml(), true);
        var retry = banner.querySelector("#gpa-retry");
        if (retry) retry.addEventListener("click", connect);
        return;
      }
      if (!res.models.length) {
        connected = false;
        setStatus("off");
        var bb = showBanner("Ollama is running but has no chat models. Pull one:" +
          "<code>ollama pull " + DEFAULT_MODEL + "</code>" +
          "<button class=\"gpa-btn gpa-secondary\" id=\"gpa-retry\">Retry</button>", true);
        var r2 = bb.querySelector("#gpa-retry");
        if (r2) r2.addEventListener("click", connect);
        return;
      }
      connected = true;
      setStatus("ok");
      populateModelSelect();
      var ex = els.messages.querySelector(".gpa-banner");
      if (ex) ex.remove();
      els.footer.style.display = "block";
      if (!els.messages.querySelector(".gpa-msg")) {
        var hint = el("div", "gpa-msg gpa-bot",
          "Hi! Ask me about GEM-pRF — installation, configuration parameters, running the software, or the paper. " +
          "I answer only from the official GEM-pRF sources, using <strong>" + escapeHtml(chatModel) +
          "</strong> on your machine.");
        els.messages.appendChild(hint);
      }
      els.input.focus();
      warnIfWeakModel();

      embedModel = pickEmbedModel(embedModels);
      chunkVectors = null;
      paramRows = null;
      if (!embedModel) return;
      return buildChunkVectors().then(function () {
        return buildParamVectors();
      }).catch(function () {
        embedModel = null;
        chunkVectors = null;
        paramRows = null;
      });
    });
  }

  // Build the widget DOM and wire up its events.
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
          '<textarea id="gpa-input" rows="1" placeholder="Ask about GEM-pRF…"></textarea>' +
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

  var opened = false;
  // Open the panel; load the index and connect on first open.
  function openPanel() {
    els.panel.classList.add("gpa-open");
    els.launcher.classList.add("gpa-hidden");
    if (!opened) {
      opened = true;
      ensureIndex().then(connect);
    }
  }
  // Close the panel.
  function closePanel() {
    els.panel.classList.remove("gpa-open");
    els.launcher.classList.remove("gpa-hidden");
  }

  var indexPromise = null;
  // Fetch and prepare the knowledge index once.
  function ensureIndex() {
    if (indexPromise) return indexPromise;
    indexPromise = fetch(INDEX_URL, { cache: "no-cache" }).then(function (r) { return r.json(); }).then(function (j) {
      idx = j;
      bm25 = prepareBm25(idx.chunks);
    }).catch(function (e) {
      showBanner("Failed to load the knowledge index: " + (e.message || e), true);
      throw e;
    });
    return indexPromise;
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", buildPanel);
    } else {
      buildPanel();
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      tokenize: tokenize, renderMarkdown: renderMarkdown,
      matchParametersEmbedding: matchParametersEmbedding, matchParametersSubstring: matchParametersSubstring,
      buildParamVectors: buildParamVectors, retrieve: retrieve, buildHumanPrompt: buildHumanPrompt,
      termFork: termFork, forkQuestion: forkQuestion, CLARIFY_DIRECTIVE: CLARIFY_DIRECTIVE,
      isCodeRequest: isCodeRequest, codeParamForQuestion: codeParamForQuestion, findCodeChunk: findCodeChunk, findCodeChunks: findCodeChunks,
      namesParameter: namesParameter, valueClarifyingQuestion: valueClarifyingQuestion,
      needsGoal: needsGoal, engageWithoutEvidence: engageWithoutEvidence,
      INTERACTIVE_NO_EVIDENCE_MESSAGE: INTERACTIVE_NO_EVIDENCE_MESSAGE,
      streamChat: streamChat, SYSTEM_PROMPT: SYSTEM_PROMPT,
      INSUFFICIENT_EVIDENCE_MESSAGE: INSUFFICIENT_EVIDENCE_MESSAGE,
      contextualize: contextualize, chatComplete: chatComplete,
      isVerificationFollowup: isVerificationFollowup, verifyPriorAnswer: verifyPriorAnswer,
      adjudicateConsistency: adjudicateConsistency, composeVerify: composeVerify, firstSentence: firstSentence,
      analyzeFold: analyzeFold, generateAnswer: generateAnswer,
      _setHistory: function (h) { history = h; },
      pickEmbedModel: pickEmbedModel, buildChunkVectors: buildChunkVectors, ollamaEmbed: ollamaEmbed,
      _load: function (j) { idx = j; bm25 = prepareBm25(idx.chunks); _tokenOwnersCache = null; _anchorVocabCache = null; },
      _setModel: function (m) { chatModel = m; },
      _setEmbedModel: function (m) { embedModel = m; },
      _state: function () { return { embedModel: embedModel, haveVectors: !!chunkVectors, chunks: idx.chunks.length }; }
    };
  }
})();
