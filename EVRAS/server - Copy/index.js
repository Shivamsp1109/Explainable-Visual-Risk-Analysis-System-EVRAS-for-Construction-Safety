const cors = require("cors");
const dotenv = require("dotenv");
const express = require("express");
const fs = require("fs");
const multer = require("multer");
const path = require("path");
const { spawn } = require("child_process");

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const CORS_ORIGIN = process.env.CORS_ORIGIN || "http://localhost:5173";

const ROOT = path.resolve(__dirname, "..");
const UPLOAD_DIR = path.join(__dirname, "uploads");
const RESULT_DIR = path.join(__dirname, "results");
const WEB_DIST = path.join(ROOT, "web", "dist");

const ensureDir = (dirPath) => {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
};

ensureDir(UPLOAD_DIR);
ensureDir(RESULT_DIR);

app.use(cors({ origin: CORS_ORIGIN }));
app.use("/results", express.static(RESULT_DIR));

if (fs.existsSync(WEB_DIST)) {
  app.use(express.static(WEB_DIST));
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || ".jpg";
    const stamp = Date.now();
    const rand = Math.random().toString(36).slice(2, 8);
    cb(null, `${stamp}_${rand}${ext}`);
  }
});

const upload = multer({ storage });

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

app.post("/api/analyze", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ ok: false, error: "Image file is required." });
  }

  const includeLlm = String(req.body.include_llm || "true").toLowerCase() !== "false";
  const baseName = path.parse(req.file.filename).name;

  const outputImage = path.join(RESULT_DIR, `${baseName}_boxed.jpg`);
  const outputJson = path.join(RESULT_DIR, `${baseName}_result.json`);
  const scriptPath = path.join(ROOT, "scripts", "run_pipeline.py");

  const args = [
    scriptPath,
    "--image", req.file.path,
    "--output-image", outputImage,
    "--output-json", outputJson,
    "--include-llm", includeLlm ? "true" : "false"
  ];

  const proc = spawn(PYTHON_BIN, args, { cwd: ROOT });
  let stdout = "";
  let stderr = "";

  proc.stdout.on("data", (data) => {
    stdout += data.toString();
  });

  proc.stderr.on("data", (data) => {
    stderr += data.toString();
  });

  proc.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({
        ok: false,
        error: "Pipeline failed",
        details: stderr || "Unknown error"
      });
    }

    let result;
    try {
      const jsonText = fs.readFileSync(outputJson, "utf-8");
      result = JSON.parse(jsonText);
    } catch (err) {
      return res.status(500).json({
        ok: false,
        error: "Failed to load pipeline output JSON",
        details: err.message || stderr || "Unknown error"
      });
    }

    const imageUrl = `/results/${path.basename(outputImage)}`;
    const jsonUrl = `/results/${path.basename(outputJson)}`;

    return res.json({
      ok: true,
      result,
      image_url: imageUrl,
      download_url: imageUrl,
      json_url: jsonUrl
    });
  });
});

app.listen(PORT, () => {
  console.log(`EVRAS server running on http://localhost:${PORT}`);
});
