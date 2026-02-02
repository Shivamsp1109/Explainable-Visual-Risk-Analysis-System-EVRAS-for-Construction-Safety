import React, { useMemo, useState } from "react";

const formatJson = (value) => {
  if (!value) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const buildSimpleInference = (jsonData) => {
  if (!jsonData) {
    return ["No analysis data available yet."];
  }

  const lines = [];
  const level = jsonData.risk_level;
  if (level) {
    if (level === "HIGH") {
      lines.push("Overall risk is high and needs immediate attention.");
    } else if (level === "MEDIUM") {
      lines.push("Overall risk is moderate and should be addressed soon.");
    } else {
      lines.push("Overall risk is low based on the current scene.");
    }
  }

  const ppe = jsonData.ppe_compliance;
  if (ppe && ppe.score !== null && ppe.score !== undefined) {
    const missing = [];
    if (ppe.helmet === false) missing.push("helmet");
    if (ppe.vest === false) missing.push("safety vest");
    if (ppe.mask === false) missing.push("mask");

    if (missing.length > 0) {
      lines.push(`Some people appear to be missing ${missing.join(", ")}.`);
    } else if (ppe.helmet && ppe.vest && ppe.mask) {
      lines.push("PPE appears to be present for all detected people.");
    }
  }

  const factors = Array.isArray(jsonData.factors) ? jsonData.factors : [];
  if (factors.length > 0) {
    lines.push("Key risks detected include:");
    factors.slice(0, 5).forEach((factor) => {
      const cleaned = String(factor).replace(/\(p=.*?\)/gi, "").trim();
      lines.push(`- ${cleaned}`);
    });
  }

  if (lines.length === 0) {
    lines.push("No significant risks were identified in this image.");
  }

  return lines;
};

const App = () => {
  const [file, setFile] = useState(null);
  const [includeLlm, setIncludeLlm] = useState(true);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [outputImageUrl, setOutputImageUrl] = useState("");
  const [jsonUrl, setJsonUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [explainData, setExplainData] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState("");

  const isExplainPage = typeof window !== "undefined" && window.location.hash.startsWith("#/explain");
  const explainJsonUrl = isExplainPage
    ? new URLSearchParams(window.location.hash.split("?")[1] || "").get("json")
    : null;

  const jsonText = useMemo(() => formatJson(result), [result]);

  const handleFileChange = (event) => {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    setResult(null);
    setOutputImageUrl("");
    setJsonUrl("");
    setError("");

    if (nextFile) {
      const url = URL.createObjectURL(nextFile);
      setPreviewUrl(url);
    } else {
      setPreviewUrl("");
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please select an image.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("image", file);
    formData.append("include_llm", includeLlm ? "true" : "false");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Request failed");
      }

      setResult(data.result);
      setOutputImageUrl(data.image_url || "");
      setJsonUrl(data.json_url || "");
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const loadExplain = async () => {
    if (!explainJsonUrl) {
      setExplainError("No JSON file selected.");
      return;
    }
    setExplainLoading(true);
    setExplainError("");
    try {
      const response = await fetch(explainJsonUrl);
      const data = await response.json();
      setExplainData(data);
    } catch (err) {
      setExplainError("Failed to load inference data.");
    } finally {
      setExplainLoading(false);
    }
  };

  if (isExplainPage) {
    const inferenceLines = buildSimpleInference(explainData);
    return (
      <div className="app">
        <header className="hero">
          <div>
            <p className="eyebrow">EVRAS Risk Summary</p>
            <h1>Simple Inference</h1>
            <p className="subtitle">
              A plain-language explanation based on the JSON output. No numbers, just the story.
            </p>
          </div>
          <a className="chip button-chip" href="/#">
            Back to Upload
          </a>
        </header>

        <section className="panel">
          {!explainData && (
            <button className="cta" onClick={loadExplain} disabled={explainLoading}>
              {explainLoading ? "Loading..." : "Load Inference"}
            </button>
          )}
          {explainError && <p className="error">{explainError}</p>}

          {explainData && (
            <div className="inference">
              {inferenceLines.map((line, index) => (
                <p key={`${index}-${line}`}>{line}</p>
              ))}
            </div>
          )}
        </section>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">EVRAS Risk Analyzer</p>
          <h1>Visual PPE &amp; Risk Scan</h1>
          <p className="subtitle">
            Upload a construction or industrial image, inspect bounding boxes,
            and receive a structured risk report in seconds.
          </p>
        </div>
        <div className="chip">Inference Ready</div>
      </header>

      <section className="panel">
        <form onSubmit={handleSubmit} className="form">
          <label className="file-input">
            <span>Input Image</span>
            <input type="file" accept="image/*" onChange={handleFileChange} />
          </label>

          <div className="toggle-row">
            <button
              type="button"
              className={`toggle ${includeLlm ? "on" : "off"}`}
              onClick={() => setIncludeLlm((prev) => !prev)}
            >
              <span className="toggle-knob" />
            </button>
            <div>
              <p className="toggle-title">LLM Explanation</p>
              <p className="toggle-subtitle">
                On by default. Disable to skip Ollama explanation.
              </p>
            </div>
          </div>

          <button className="cta" type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
          {error && <p className="error">{error}</p>}
        </form>

        <div className="preview">
          <div className="preview-card">
            <h3>Input Preview</h3>
            {previewUrl ? (
              <img src={previewUrl} alt="Input preview" />
            ) : (
              <div className="placeholder">No image selected</div>
            )}
          </div>

          <div className="preview-card">
            <h3>Output Preview</h3>
            {outputImageUrl ? (
              <img src={outputImageUrl} alt="Output preview" />
            ) : (
              <div className="placeholder">Awaiting analysis</div>
            )}
            {outputImageUrl && (
              <a className="download" href={outputImageUrl} download>
                Download Output Image
              </a>
            )}
            {jsonUrl && (
              <a className="download secondary" href={`/#/explain?json=${encodeURIComponent(jsonUrl)}`}>
                Open Simple Inference
              </a>
            )}
          </div>
        </div>
      </section>

      <section className="panel grid">
        <div className="json-block">
          <div className="json-header">
            <h3>Result JSON</h3>
            {jsonUrl && (
              <a className="download tiny" href={jsonUrl} download>
                Download JSON
              </a>
            )}
          </div>
          <pre>{jsonText || "No result yet."}</pre>
        </div>
        <div className="stats-block">
          <h3>Output Highlights</h3>
          {result ? (
            <ul>
              <li>
                <span>Risk Level</span>
                <strong>{result.risk_level}</strong>
              </li>
              <li>
                <span>Confidence</span>
                <strong>{result.confidence}</strong>
              </li>
              <li>
                <span>PPE Compliance</span>
                <strong>{result.ppe_compliance?.score ?? "N/A"}</strong>
              </li>
              <li>
                <span>Top Factors</span>
                <strong>{result.factors?.length || 0}</strong>
              </li>
            </ul>
          ) : (
            <p className="placeholder">Upload an image to view metrics.</p>
          )}
        </div>
      </section>
    </div>
  );
};

export default App;
