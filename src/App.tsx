import { useState, useEffect, useRef, useCallback, type RefObject } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";

// Local border surveillance footage (Getty Images)
import vid1 from "@/imports/gettyimages-1144763193-640_adpp.mp4";
import vid2 from "@/imports/gettyimages-1217624486-640_adpp.mp4";
import vid3 from "@/imports/gettyimages-2060388063-640_adpp.mp4";
import vid4 from "@/imports/gettyimages-2215078451-640_adpp.mp4";

// ─── COCO-SSD singleton ───────────────────────────────────────────────────────
// Loads once, shared across all camera feeds.
let _modelPromise: Promise<cocoSsd.ObjectDetection> | null = null;
function loadModel() {
  if (!_modelPromise) {
    _modelPromise = cocoSsd.load({ base: "lite_mobilenet_v2" });
  }
  return _modelPromise;
}

function drawDetections(
  ctx: CanvasRenderingContext2D,
  preds: cocoSsd.DetectedObject[],
  isNight: boolean
) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (const p of preds) {
    if (p.score < 0.45) continue;
    if (p.class !== "person") continue;
    const [x, y, w, h] = p.bbox;
    const isPerson = true;
    const col = isNight ? "#4ade80" : "#3b82f6";

    // Box
    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.8;
    ctx.shadowColor = col;
    ctx.shadowBlur = 6;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();

    // Corner ticks
    const tk = 10;
    ctx.strokeStyle = col;
    ctx.lineWidth = 2.5;
    for (const [cx, cy, sx, sy] of [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]] as [number,number,number,number][]) {
      ctx.beginPath(); ctx.moveTo(cx, cy + sy*tk); ctx.lineTo(cx, cy); ctx.lineTo(cx + sx*tk, cy); ctx.stroke();
    }

    // Label pill
    const label = `${isPerson ? "PERSON" : p.class.toUpperCase()}  ${(p.score*100).toFixed(0)}%`;
    ctx.font = "bold 10px 'DM Mono', monospace";
    const tw = ctx.measureText(label).width;
    const lx = Math.max(x, 2);
    const ly = y > 22 ? y - 6 : y + h + 18;
    ctx.fillStyle = col + "cc";
    ctx.beginPath();
    ctx.roundRect(lx, ly - 14, tw + 10, 17, 3);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.fillText(label, lx + 5, ly);

    // Pulsing dot
    ctx.beginPath();
    ctx.arc(lx + 4, ly - 7, 3, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
  }
}

// Hook: runs COCO-SSD on a video element, drawing results onto a canvas.
// intervalMs controls how often inference fires (lower = faster but heavier).
function useVideoDetection(
  videoRef: RefObject<HTMLVideoElement | null>,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  enabled: boolean,
  intervalMs: number,
  isNight: boolean
) {
  useEffect(() => {
    if (!enabled) {
      const canvas = canvasRef.current;
      if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    let active = true;
    let rafId: number;
    let lastRun = 0;
    let running = false;

    loadModel().then((model) => {
      const loop = async (time: number) => {
        if (!active) return;
        if (time - lastRun > intervalMs && !running) {
          lastRun = time;
          running = true;
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video && canvas && video.readyState >= 2 && video.videoWidth > 0) {
            try {
              canvas.width  = video.videoWidth;
              canvas.height = video.videoHeight;
              const preds = await model.detect(video);
              if (active) {
                const ctx = canvas.getContext("2d");
                if (ctx) drawDetections(ctx, preds, isNight);
              }
            } catch (_) { /* video may not be ready yet */ }
          }
          running = false;
        }
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    });

    return () => {
      active = false;
      cancelAnimationFrame(rafId);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, isNight]);
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface Alert {
  id: string;
  type: "fence_crossing" | "anpr" | "face" | "night" | "motion";
  severity: "critical" | "warning" | "info";
  camera_id: string;
  message: string;
  ts: string;
  plate?: string;
  confidence?: number;
  acked: boolean;
}

interface ANPRScan {
  id: string;
  plate: string;
  conf: number;
  camera_id: string;
  ts: string;
  status: "watchlist" | "clear" | "unknown";
  region: string;
  state: "scanning" | "done";
}

interface FaceScan {
  id: string;
  label: string;
  conf: number;
  camera_id: string;
  ts: string;
  status: "identified" | "unknown" | "watchlist";
  state: "scanning" | "done";
  avatar: string;
}

interface NightStatus {
  camera_id: string;
  label: string;
  irOn: boolean;
  lux: number;
  tempC: number;
  motionScore: number;
}

// ─── Camera config ────────────────────────────────────────────────────────────

interface CamConfig {
  id: string;
  label: string;
  sector: string;
  coords: string;
  isNight: boolean;
  isRecording: boolean;
  online: boolean;
  fps: number;
  videoUrl: string;
  poster: string;
}

const CAMERAS: CamConfig[] = [
  {
    id: "CAM-01", label: "North Gate — Alpha",  sector: "Sector 1",
    coords: "34.0837°N 74.7973°E", isNight: false, isRecording: true, online: true, fps: 30,
    videoUrl: vid1, poster: "",
  },
  {
    id: "CAM-02", label: "East Perimeter",       sector: "Sector 2",
    coords: "34.1526°N 74.8412°E", isNight: true, isRecording: true, online: true, fps: 25,
    videoUrl: vid2, poster: "",
  },
  {
    id: "CAM-03", label: "Checkpoint Bravo",     sector: "Sector 3",
    coords: "33.9811°N 74.7201°E", isNight: false, isRecording: false, online: true, fps: 30,
    videoUrl: vid3, poster: "",
  },
  {
    id: "CAM-04", label: "River Crossing",       sector: "Sector 4",
    coords: "33.8942°N 74.8672°E", isNight: true, isRecording: true, online: true, fps: 15,
    videoUrl: vid4, poster: "",
  },
  {
    id: "CAM-05", label: "South Patrol Road",    sector: "Sector 5",
    coords: "33.7489°N 74.7104°E", isNight: false, isRecording: false, online: true, fps: 30,
    videoUrl: vid2, poster: "", // reuse vid2 for 5th camera
  },
  {
    id: "CAM-06", label: "Watchtower WK-7",      sector: "Sector 6",
    coords: "34.2201°N 74.9890°E", isNight: false, isRecording: false, online: false, fps: 0,
    videoUrl: "", poster: "",
  },
];


// ─── Data pools ───────────────────────────────────────────────────────────────

const PLATE_POOL: Omit<ANPRScan, "id" | "ts" | "state">[] = [
  { plate: "RJ14-CD-5678", conf: 97.4, camera_id: "CAM-03", status: "watchlist", region: "Rajasthan, IND" },
  { plate: "PB-03-AK-1122", conf: 94.1, camera_id: "CAM-01", status: "clear",     region: "Punjab, IND"    },
  { plate: "LAH-PKX-449",   conf: 88.6, camera_id: "CAM-03", status: "unknown",   region: "Unknown"        },
  { plate: "HR-26-CN-7745", conf: 99.2, camera_id: "CAM-01", status: "clear",     region: "Haryana, IND"   },
  { plate: "JK-02-PK-3310", conf: 91.8, camera_id: "CAM-03", status: "watchlist", region: "J&K, IND"       },
  { plate: "DL-01-AB-4421", conf: 96.3, camera_id: "CAM-01", status: "clear",     region: "Delhi, IND"     },
  { plate: "ISB-MXZ-887",   conf: 83.2, camera_id: "CAM-03", status: "unknown",   region: "Unknown"        },
  { plate: "KPK-0049-LMN",  conf: 79.5, camera_id: "CAM-03", status: "watchlist", region: "Unknown"        },
];

const FACE_POOL: Omit<FaceScan, "id" | "ts" | "state">[] = [
  { label: "UNKNOWN #A7F2",             conf: 0,    camera_id: "CAM-01", status: "unknown",    avatar: "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=64&h=64&fit=crop" },
  { label: "Ahmed Karimi",              conf: 91.3, camera_id: "CAM-03", status: "watchlist",  avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=64&h=64&fit=crop" },
  { label: "Authorized — Gate Officer", conf: 98.7, camera_id: "CAM-01", status: "identified", avatar: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=64&h=64&fit=crop" },
  { label: "UNKNOWN #B3D9",             conf: 0,    camera_id: "CAM-02", status: "unknown",    avatar: "https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=64&h=64&fit=crop" },
  { label: "Rajiv Mehta",               conf: 95.1, camera_id: "CAM-01", status: "identified", avatar: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=64&h=64&fit=crop" },
  { label: "Zubair Khan",               conf: 87.4, camera_id: "CAM-03", status: "watchlist",  avatar: "https://images.unsplash.com/photo-1519345182560-3f2917c472ef?w=64&h=64&fit=crop" },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pad(n: number) { return String(n).padStart(2, "0"); }
function ts(d: Date) { return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; }
function useNow() {
  const [t, setT] = useState(new Date());
  useEffect(() => { const id = setInterval(() => setT(new Date()), 1000); return () => clearInterval(id); }, []);
  return t;
}
function greeting(h: number) { return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening"; }

// Clamp x to [lo, hi]
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }

// ─── Icons ────────────────────────────────────────────────────────────────────
const Icon = {
  Camera: () => (<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M1 5.5A1.5 1.5 0 012.5 4h1.382l.894-1.789A.5.5 0 015.224 2h5.553a.5.5 0 01.447.276L12.118 4H13.5A1.5 1.5 0 0115 5.5v7A1.5 1.5 0 0113.5 14h-11A1.5 1.5 0 011 12.5v-7z" stroke="currentColor" strokeWidth="1.2"/><circle cx="8" cy="9" r="2.2" stroke="currentColor" strokeWidth="1.2"/></svg>),
  Alert:  () => (<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2L14 13H2L8 2z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/><line x1="8" y1="6" x2="8" y2="9.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/><circle cx="8" cy="11.5" r="0.6" fill="currentColor"/></svg>),
  Plate:  () => (<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="1" y="4.5" width="14" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.2"/><rect x="3" y="6.5" width="10" height="3" rx="0.5" fill="currentColor" opacity="0.15"/><circle cx="4" cy="8" r="0.8" fill="currentColor"/><circle cx="12" cy="8" r="0.8" fill="currentColor"/></svg>),
  Face:   () => (<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="6" r="2.8" stroke="currentColor" strokeWidth="1.2"/><path d="M2.5 14c0-3.038 2.462-5.5 5.5-5.5S13.5 10.962 13.5 14" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>),
  Moon:   () => (<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M10 7.5A4 4 0 016 3.5a4 4 0 00.5 8A4 4 0 0010 7.5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/></svg>),
  Sun:    () => (<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="2.5" stroke="currentColor" strokeWidth="1.2"/><line x1="7" y1="1" x2="7" y2="3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/><line x1="7" y1="11" x2="7" y2="13" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/><line x1="1" y1="7" x2="3" y2="7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/><line x1="11" y1="7" x2="13" y2="7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>),
  Grid:   () => (<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2"/><rect x="8" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2"/><rect x="1" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2"/><rect x="8" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2"/></svg>),
  Box:    () => (<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="3" width="12" height="8" rx="1" stroke="currentColor" strokeWidth="1.3" strokeDasharray="2 1.5"/><path d="M4 3V1.5M10 3V1.5M4 11v1.5M10 11v1.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>),
  Eye:    () => (<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M1 7s2.5-4.5 6-4.5S13 7 13 7s-2.5 4.5-6 4.5S1 7 1 7z" stroke="currentColor" strokeWidth="1.2"/><circle cx="7" cy="7" r="1.8" stroke="currentColor" strokeWidth="1.2"/></svg>),
  Thermo: () => (<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M6 1v5.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/><circle cx="6" cy="9" r="2" stroke="currentColor" strokeWidth="1.2"/></svg>),
  Snapshot: () => (<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 9.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" stroke="currentColor" strokeWidth="1.2"/><path d="M2 4.5h1.5L4.5 3h5l1 1.5H12a1 1 0 011 1v5a1 1 0 01-1 1H2a1 1 0 01-1-1v-5a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2"/></svg>),
};

function Dot({ color, pulse }: { color: string; pulse?: boolean }) {
  return <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${color} ${pulse ? "dot-pulse" : ""}`} />;
}

// ─── Camera Feed ─────────────────────────────────────────────────────────────

function CameraFeed({
  cam, selected, onClick, compact, showBoxes, isSelected, snapshotRef,
}: {
  cam: CamConfig; selected: boolean;
  onClick: () => void; compact?: boolean;
  showBoxes: boolean; isSelected: boolean;
  snapshotRef?: React.MutableRefObject<(() => void) | null>;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!snapshotRef) return;
    snapshotRef.current = () => {
      const video = videoRef.current;
      if (!video) return;
      const w = video.videoWidth || video.clientWidth;
      const h = video.videoHeight || video.clientHeight;
      const snap = document.createElement("canvas");
      snap.width = w; snap.height = h;
      const ctx = snap.getContext("2d")!;
      ctx.drawImage(video, 0, 0, w, h);
      // overlay detection boxes if visible
      if (canvasRef.current) ctx.drawImage(canvasRef.current, 0, 0, w, h);
      snap.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = `IBVAP_${cam.id}_${Date.now()}.png`;
        a.click(); URL.revokeObjectURL(url);
      }, "image/png");
    };
    return () => { if (snapshotRef) snapshotRef.current = null; };
  }, [cam.id, snapshotRef]);

  useEffect(() => { videoRef.current?.play().catch(() => {}); }, [cam.id]);

  // Run real COCO-SSD detection:
  // selected camera → 100 ms interval (~10 fps inference)
  // thumbnails       → 2000 ms interval (0.5 fps, low cost)
  useVideoDetection(
    videoRef,
    canvasRef,
    showBoxes && cam.online && !!cam.videoUrl,
    isSelected ? 100 : 2000,
    cam.isNight
  );

  return (
    <div
      onClick={onClick}
      className={`relative overflow-hidden rounded-xl cursor-pointer select-none transition-all duration-200
        ${!cam.online ? "opacity-40" : ""}
        ${selected
          ? "ring-2 ring-blue-500 ring-offset-2 ring-offset-white"
          : "ring-1 ring-black/8 hover:ring-blue-300 hover:ring-offset-1 hover:ring-offset-white"
        }`}
      style={{ aspectRatio: "16/10", background: "#060d18" }}
    >
      {cam.online && cam.videoUrl ? (
        <video
          ref={videoRef}
          src={cam.videoUrl}
          poster={cam.poster}
          className={`absolute inset-0 w-full h-full object-cover ${cam.isNight ? "brightness-50 saturate-0" : ""}`}
          autoPlay muted loop playsInline
          style={{ pointerEvents: "none" }}
        />
      ) : !cam.online ? null : (
        <img src={cam.poster} alt={cam.label} className="absolute inset-0 w-full h-full object-cover" />
      )}

      {/* COCO-SSD canvas — sits directly over video, pointer-events none */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ pointerEvents: "none" }}
      />

      {/* Night green tint */}
      {cam.isNight && cam.online && (
        <div className="absolute inset-0" style={{ background: "rgba(34,197,94,0.12)", mixBlendMode: "screen" }} />
      )}

      {/* Grid overlay */}
      <div className="absolute inset-0 cam-grid-bg" />

      {/* Scan line */}
      {cam.online && (
        <div className={`scanline ${cam.isNight ? "night-scanline" : ""}`} style={{ top: "30%" }} />
      )}

      {/* Detection drawn on canvas by COCO-SSD — no div boxes needed */}

      {/* Top-left: cam ID + NV badge */}
      <div className="absolute top-2 left-2 flex items-center gap-1.5 z-10">
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "rgba(255,255,255,0.75)", textShadow: "0 1px 3px #000" }}>
          {cam.id}
        </span>
        {cam.isNight && (
          <span className="text-emerald-300 bg-emerald-900/60 border border-emerald-500/40 rounded px-1 text-[9px]" style={{ fontFamily: "var(--font-mono)" }}>NV</span>
        )}
      </div>

      {/* REC */}
      {cam.isRecording && cam.online && (
        <div className="absolute top-2 right-2 flex items-center gap-1 z-10">
          <span className="w-2 h-2 rounded-full bg-red-500 dot-pulse" />
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "#ef4444" }}>REC</span>
        </div>
      )}

      {/* Corner marks */}
      <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-white/20 rounded-tl-xl" />
      <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-white/20 rounded-tr-xl" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b border-l border-white/20 rounded-bl-xl" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-white/20 rounded-br-xl" />

      {/* Live detection badge */}
      {showBoxes && cam.online && (
        <div className="absolute bottom-2 left-2 z-10 flex items-center gap-1 bg-blue-500/20 border border-blue-400/40 rounded px-1.5 py-0.5">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 dot-pulse" />
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "#93c5fd" }}>DETECTING</span>
        </div>
      )}

      {/* FPS */}
      {!compact && cam.online && (
        <div className="absolute bottom-2 right-2 z-10">
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "rgba(255,255,255,0.35)" }}>{cam.fps} FPS</span>
        </div>
      )}

      {/* Offline */}
      {!cam.online && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80">
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "#ef4444" }}>NO SIGNAL</span>
        </div>
      )}
    </div>
  );
}

// ─── ANPR Panel ───────────────────────────────────────────────────────────────

function ConfidenceTick({ target }: { target: number }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let v = 0; const step = target / 28;
    const id = setInterval(() => { v = Math.min(v + step, target); setVal(v); if (v >= target) clearInterval(id); }, 35);
    return () => clearInterval(id);
  }, [target]);
  return <p className="text-blue-400 font-bold text-sm" style={{ fontFamily: "var(--font-mono)" }}>{val.toFixed(1)}%</p>;
}

function ANPRPanel({ scans }: { scans: ANPRScan[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const scanning = scans.find(s => s.state === "scanning");
  const sty = { watchlist: "bg-red-100 text-red-700 border-red-200 dot-bg-red-500", clear: "bg-emerald-100 text-emerald-700 border-emerald-200", unknown: "bg-amber-100 text-amber-700 border-amber-200" };
  const dot = { watchlist: "bg-red-500", clear: "bg-emerald-500", unknown: "bg-amber-500" };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-100 flex-shrink-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-slate-700"><Icon.Plate /><span className="text-xs font-semibold">ANPR Engine</span></div>
          <div className="flex items-center gap-1.5 bg-emerald-50 border border-emerald-200 rounded-full px-2 py-0.5">
            <Dot color="bg-emerald-500" pulse /><span className="text-[10px] text-emerald-700 font-medium" style={{ fontFamily: "var(--font-mono)" }}>LIVE</span>
          </div>
        </div>
        {scanning ? (
          <div className="bg-slate-900 rounded-xl p-3 relative overflow-hidden" style={{ minHeight: 80 }}>
            <div className="absolute inset-0 overflow-hidden rounded-xl"><div className="scanline" /></div>
            <div className="relative z-10">
              <p className="text-[10px] text-slate-400 mb-1" style={{ fontFamily: "var(--font-mono)" }}>SCANNING PLATE</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-slate-800 rounded-lg h-9 flex items-center px-3 border border-slate-700">
                  <span className="font-bold tracking-[0.25em] text-sm" style={{ fontFamily: "var(--font-mono)" }}>
                    {scanning.plate.split("").map((c, i) => (
                      <span key={i} className={i < 4 ? "text-blue-400" : "text-white"}>{c}</span>
                    ))}
                  </span>
                </div>
                <div className="text-right"><p className="text-[9px] text-slate-500" style={{ fontFamily: "var(--font-mono)" }}>CONF</p><ConfidenceTick target={scanning.conf} /></div>
              </div>
              <div className="mt-2 flex gap-0.5">
                {[...Array(20)].map((_, i) => <div key={i} className="flex-1 h-1 rounded-full bg-blue-500/20 overflow-hidden"><div className="h-full bg-blue-500 animate-pulse" style={{ animationDelay: `${i * 55}ms`, width: i < 13 ? "100%" : "0" }} /></div>)}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-slate-200 flex items-center justify-center text-slate-400"><Icon.Plate /></div>
            <div><p className="text-xs font-medium text-slate-600">Ready to scan</p><p className="text-[10px] text-slate-400" style={{ fontFamily: "var(--font-mono)" }}>Monitoring {CAMERAS.filter(c => c.online).length} cameras</p></div>
          </div>
        )}
        <div className="flex gap-2 mt-3">
          {[["Total", scans.length, "text-slate-700"],["Watchlist", scans.filter(s=>s.status==="watchlist").length,"text-red-600"],["Clear",scans.filter(s=>s.status==="clear").length,"text-emerald-600"],["Unknown",scans.filter(s=>s.status==="unknown").length,"text-amber-600"]].map(([l, v, c]) => (
            <div key={l as string} className="flex-1 text-center bg-slate-50 rounded-lg py-1.5">
              <p className={`text-sm font-bold ${c}`}>{v}</p><p className="text-[9px] text-slate-400">{l}</p>
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
        {scans.map(s => (
          <div key={s.id} onClick={() => setExpanded(expanded === s.id ? null : s.id)} className={`rounded-xl border cursor-pointer transition-all overflow-hidden ${s.state === "scanning" ? "bg-blue-50 border-blue-200" : "bg-white border-slate-200 hover:border-slate-300 hover:shadow-sm"}`}>
            <div className="flex items-center gap-3 p-3">
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot[s.status]}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-900 tracking-wider text-xs" style={{ fontFamily: "var(--font-mono)" }}>{s.plate}</span>
                  {s.state === "scanning" && <span className="text-[9px] text-blue-500 animate-pulse" style={{ fontFamily: "var(--font-mono)" }}>SCANNING…</span>}
                </div>
                <p className="text-[10px] text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-mono)" }}>{s.region} · {s.camera_id} · {s.ts}</p>
              </div>
              <div className="flex flex-col items-end gap-1">
                <span className={`text-[9px] font-semibold border rounded-full px-1.5 py-px uppercase ${sty[s.status]}`}>{s.status}</span>
                <span className="text-[9px] text-slate-400" style={{ fontFamily: "var(--font-mono)" }}>{s.conf.toFixed(1)}%</span>
              </div>
            </div>
            {expanded === s.id && s.state === "done" && (
              <div className="border-t border-slate-100 bg-slate-50 px-3 py-2.5 grid grid-cols-2 gap-x-4 gap-y-1.5">
                {[["Camera",s.camera_id],["Region",s.region],["Confidence",`${s.conf.toFixed(1)}%`],["Time",s.ts],["Status",s.status.toUpperCase()]].map(([k,v])=>(<div key={k as string}><p className="text-[9px] text-slate-400 uppercase tracking-widest">{k}</p><p className="text-[10px] text-slate-700 font-medium" style={{fontFamily:"var(--font-mono)"}}>{v}</p></div>))}
                {s.status === "watchlist" && <div className="col-span-2 mt-1 bg-red-50 border border-red-200 rounded-lg p-2"><p className="text-[10px] text-red-700 font-semibold">⚠ Watchlist vehicle — alert dispatched to Control Room</p></div>}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Face Panel ───────────────────────────────────────────────────────────────

function FacePanel({ scans }: { scans: FaceScan[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const scanning = scans.find(s => s.state === "scanning");
  const sty = { identified: { ring: "ring-emerald-400", badge: "bg-emerald-100 text-emerald-700 border-emerald-200" }, watchlist: { ring: "ring-red-400", badge: "bg-red-100 text-red-700 border-red-200" }, unknown: { ring: "ring-amber-400", badge: "bg-amber-100 text-amber-700 border-amber-200" } };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-100 flex-shrink-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-slate-700"><Icon.Face /><span className="text-xs font-semibold">Face Recognition</span></div>
          <div className="flex items-center gap-1.5 bg-blue-50 border border-blue-200 rounded-full px-2 py-0.5"><Dot color="bg-blue-500" pulse /><span className="text-[10px] text-blue-700 font-medium" style={{ fontFamily: "var(--font-mono)" }}>AI ENGINE</span></div>
        </div>
        {scanning ? (
          <div className="bg-slate-900 rounded-xl p-4 relative overflow-hidden">
            <div className="absolute inset-0 overflow-hidden rounded-xl"><div className="scanline" /></div>
            <div className="relative z-10 flex items-center gap-4">
              <div className="relative flex-shrink-0">
                <div className="w-14 h-14 rounded-lg overflow-hidden border-2 border-blue-500 ring-2 ring-blue-500/30"><img src={scanning.avatar} alt="subject" className="w-full h-full object-cover" /></div>
                {[["top-0 left-0","border-t-2 border-l-2"],["top-0 right-0","border-t-2 border-r-2"],["bottom-0 left-0","border-b-2 border-l-2"],["bottom-0 right-0","border-b-2 border-r-2"]].map(([p,b])=><div key={p} className={`absolute w-3 h-3 ${p} ${b} border-blue-400`} />)}
                <div className="absolute inset-0 overflow-hidden rounded-lg"><div className="absolute left-0 right-0 h-0.5 bg-blue-400/60" style={{ animation: "scanline 1.5s linear infinite" }} /></div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[10px] text-slate-400 mb-2" style={{ fontFamily: "var(--font-mono)" }}>MATCHING EMBEDDINGS</p>
                {["Feature extraction","DB lookup","Confidence score"].map((step, i) => (
                  <div key={step} className="flex items-center gap-2 mb-1">
                    <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden"><div className="h-full bg-blue-500 rounded-full" style={{ width: i===0?"100%":i===1?"70%":"30%" }} /></div>
                    <p className="text-[9px] text-slate-500 w-24 truncate" style={{ fontFamily: "var(--font-mono)" }}>{step}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center text-slate-400"><Icon.Eye /></div>
            <div><p className="text-xs font-medium text-slate-600">Watching {CAMERAS.filter(c=>c.online).length} feeds</p><p className="text-[10px] text-slate-400" style={{ fontFamily: "var(--font-mono)" }}>ArcFace · 128-dim · MHA-DB</p></div>
          </div>
        )}
        <div className="flex gap-2 mt-3">
          {[["Scanned",scans.length,"text-slate-700"],["Identified",scans.filter(s=>s.status==="identified").length,"text-emerald-600"],["Watchlist",scans.filter(s=>s.status==="watchlist").length,"text-red-600"],["Unknown",scans.filter(s=>s.status==="unknown").length,"text-amber-600"]].map(([l,v,c])=>(
            <div key={l as string} className="flex-1 text-center bg-slate-50 rounded-lg py-1.5"><p className={`text-sm font-bold ${c}`}>{v}</p><p className="text-[9px] text-slate-400">{l}</p></div>
          ))}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
        {scans.map(s => {
          const st = sty[s.status];
          return (
            <div key={s.id} onClick={() => setExpanded(expanded === s.id ? null : s.id)} className={`rounded-xl border cursor-pointer transition-all overflow-hidden ${s.state==="scanning"?"bg-blue-50 border-blue-300":"bg-white border-slate-200 hover:border-slate-300 hover:shadow-sm"}`}>
              <div className="flex items-center gap-3 p-3">
                <div className={`relative flex-shrink-0 w-10 h-10 rounded-full ring-2 ${st.ring} overflow-hidden`}><img src={s.avatar} alt={s.label} className="w-full h-full object-cover" />{s.state==="scanning"&&<div className="absolute inset-0 bg-blue-500/20 animate-pulse" />}</div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold text-slate-800 truncate">{s.label}</p>
                  <p className="text-[10px] text-slate-400" style={{ fontFamily: "var(--font-mono)" }}>{s.camera_id} · {s.ts}</p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <span className={`text-[9px] font-bold border rounded-full px-1.5 py-px ${st.badge}`}>{s.status}</span>
                  {s.conf > 0 && <span className="text-[9px] text-slate-400" style={{ fontFamily: "var(--font-mono)" }}>{s.conf.toFixed(1)}%</span>}
                </div>
              </div>
              {expanded === s.id && s.state === "done" && (
                <div className="border-t border-slate-100 bg-slate-50 px-3 py-2.5">
                  <div className="flex items-start gap-3">
                    <img src={s.avatar} alt="" className="w-12 h-12 rounded-lg object-cover flex-shrink-0 border border-slate-200" />
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 flex-1">
                      {[["Camera",s.camera_id],["Time",s.ts],["Confidence",s.conf>0?`${s.conf.toFixed(1)}%`:"N/A"],["Model","ArcFace-128"],["DB","MHA Watchlist"],["Result",s.status.toUpperCase()]].map(([k,v])=>(<div key={k as string}><p className="text-[9px] text-slate-400 uppercase tracking-widest">{k}</p><p className="text-[10px] text-slate-700 font-medium" style={{fontFamily:"var(--font-mono)"}}>{v}</p></div>))}
                    </div>
                  </div>
                  {s.status==="watchlist"&&<div className="mt-2 bg-red-50 border border-red-200 rounded-lg p-2"><p className="text-[10px] text-red-700 font-semibold">⚠ MHA watchlist subject — alert dispatched</p></div>}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Night Panel ──────────────────────────────────────────────────────────────

function NightPanel({ nightData, onToggle }: { nightData: NightStatus[]; onToggle: (id: string) => void }) {
  const allOn = nightData.every(n => n.irOn);
  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-100 flex-shrink-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-slate-700"><Icon.Moon /><span className="text-xs font-semibold">Night Detection</span></div>
          <div className="flex items-center gap-1.5 bg-slate-900 rounded-full px-2 py-0.5"><Dot color="bg-emerald-400" pulse /><span className="text-[10px] text-emerald-400 font-medium" style={{ fontFamily: "var(--font-mono)" }}>{nightData.filter(n=>n.irOn).length} IR ACTIVE</span></div>
        </div>
        <div className="bg-slate-900 rounded-xl p-4 mb-3">
          <div className="flex items-center justify-between mb-3">
            <div><p className="text-xs font-semibold text-white">Night Vision System</p><p className="text-[10px] text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-mono)" }}>IR 850nm · YOLOv8-NightAdapt · Auto-threshold</p></div>
            <div className="flex items-center gap-2"><span className="text-[10px] text-slate-400">{allOn?"ALL ON":"PARTIAL"}</span>
              <button className={`relative w-10 h-5 rounded-full transition-colors ${allOn?"bg-emerald-500":"bg-slate-600"}`} onClick={() => nightData.forEach(n => allOn ? n.irOn && onToggle(n.camera_id) : !n.irOn && onToggle(n.camera_id))}>
                <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-all ${allOn?"left-5":"left-0.5"}`} />
              </button>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[{ icon:<Icon.Sun/>,label:"Avg Lux",val:`${Math.round(nightData.reduce((a,n)=>a+n.lux,0)/nightData.length)} lx`},{icon:<Icon.Thermo/>,label:"Avg Temp",val:`${Math.round(nightData.reduce((a,n)=>a+n.tempC,0)/nightData.length)}°C`},{icon:<Icon.Eye/>,label:"Motion",val:`${nightData.filter(n=>n.motionScore>50).length} cam`}].map(({icon,label,val})=>(
              <div key={label} className="bg-slate-800 rounded-lg p-2 text-center"><span className="text-slate-400 flex justify-center mb-1">{icon}</span><p className="text-white text-sm font-bold">{val}</p><p className="text-slate-500 text-[9px]">{label}</p></div>
            ))}
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {nightData.map(n => {
          const cam = CAMERAS.find(c => c.id === n.camera_id)!;
          const luxPct = Math.min((n.lux/200)*100,100);
          const motPct = n.motionScore;
          return (
            <div key={n.camera_id} className={`rounded-xl border overflow-hidden transition-all ${n.irOn?"border-emerald-200 bg-emerald-50":"border-slate-200 bg-white"}`}>
              <div className="flex items-center justify-between px-3 py-2.5">
                <div className="flex items-center gap-2">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${n.irOn?"bg-emerald-100 text-emerald-700":"bg-slate-100 text-slate-500"}`}><Icon.Camera /></div>
                  <div><p className="text-xs font-semibold text-slate-800">{n.camera_id}</p><p className="text-[10px] text-slate-400">{cam.label}</p></div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${n.irOn?"text-emerald-700 bg-emerald-100 border-emerald-200":"text-slate-500 bg-slate-100 border-slate-200"}`} style={{ fontFamily: "var(--font-mono)" }}>{n.irOn?"IR-ON":"IR-OFF"}</span>
                  <button onClick={()=>onToggle(n.camera_id)} className={`relative w-8 h-4 rounded-full transition-colors ${n.irOn?"bg-emerald-500":"bg-slate-300"}`}><span className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow transition-all ${n.irOn?"left-4":"left-0.5"}`} /></button>
                </div>
              </div>
              <div className="border-t border-current/10 px-3 py-2.5 grid grid-cols-3 gap-3">
                {[{label:"LUX",val:Math.round(n.lux),pct:luxPct,col:luxPct<20?"bg-emerald-500":luxPct<60?"bg-amber-400":"bg-red-400",sub:n.lux<30?"Low-light":n.lux<100?"Dusk":"Daylight"},{label:"TEMP",val:`${Math.round(n.tempC)}°C`,pct:((n.tempC+10)/60)*100,col:"bg-blue-400",sub:"Ambient"},{label:"MOTION",val:`${motPct}%`,pct:motPct,col:motPct>60?"bg-red-500":"bg-blue-400",sub:motPct>60?"Alert":"Normal"}].map(({label,val,pct,col,sub})=>(
                  <div key={label}>
                    <div className="flex justify-between mb-1"><span className="text-[9px] text-slate-500">{label}</span><span className="text-[9px] font-bold text-slate-700" style={{ fontFamily: "var(--font-mono)" }}>{val}</span></div>
                    <div className="h-1 bg-slate-200 rounded-full overflow-hidden"><div className={`h-full rounded-full transition-all ${col}`} style={{ width:`${Math.min(pct,100)}%` }} /></div>
                    <p className="text-[8px] text-slate-400 mt-0.5">{sub}</p>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
        <div className="bg-slate-900 rounded-xl p-3 mt-1">
          <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest mb-2">Night Event Log</p>
          {[{cam:"CAM-02",event:"IR activated — lux dropped below 25",t:"02:41:07"},{cam:"CAM-04",event:"Thermal signature — river bank",t:"02:38:22"},{cam:"CAM-02",event:"Motion spike 78% — east perimeter",t:"02:35:11"},{cam:"CAM-04",event:"NV calibration complete",t:"02:30:45"}].map((e,i)=>(
            <div key={i} className="flex items-start gap-2 mb-1.5"><span className="text-emerald-400 mt-0.5 flex-shrink-0">›</span><div><p className="text-[10px] text-slate-300">{e.event}</p><p className="text-[9px] text-slate-600" style={{ fontFamily: "var(--font-mono)" }}>{e.cam} · {e.t}</p></div></div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Alert detail modal ───────────────────────────────────────────────────────

function AlertModal({ a, onClose, onAck }: { a: Alert; onClose: () => void; onAck: (id: string) => void }) {
  const s = {
    critical: { bar:"bg-red-500", header:"bg-red-600", badge:"bg-red-100 text-red-700 border-red-300", label:"CRITICAL" },
    warning:  { bar:"bg-amber-400", header:"bg-amber-500", badge:"bg-amber-100 text-amber-700 border-amber-300", label:"WARNING" },
    info:     { bar:"bg-blue-400", header:"bg-blue-500", badge:"bg-blue-100 text-blue-700 border-blue-300", label:"INFO" },
  }[a.severity];
  const typeLabel = { fence_crossing:"Fence Crossing", anpr:"ANPR Detection", face:"Face Recognition", night:"Night Event", motion:"Motion Alert" }[a.type] ?? a.type;
  const emoji = a.type==="anpr"?"🚗":a.type==="face"?"👤":a.type==="night"?"🌙":a.type==="fence_crossing"?"🚨":"⚠️";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm" />
      <div className="relative w-full max-w-sm rounded-2xl overflow-hidden shadow-2xl animate-fade-up" onClick={e=>e.stopPropagation()}>
        {/* Header */}
        <div className={`${s.header} px-5 py-4 flex items-start justify-between`}>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg">{emoji}</span>
              <span className={`text-[10px] font-bold border rounded-full px-2 py-px ${s.badge}`}>{s.label}</span>
              {!a.acked && <span className="text-[10px] font-bold bg-white/20 text-white rounded-full px-2 py-px">UNACKNOWLEDGED</span>}
            </div>
            <p className="text-white font-semibold text-sm leading-snug">{a.message}</p>
          </div>
          <button onClick={onClose} className="text-white/70 hover:text-white text-lg leading-none ml-3 mt-0.5">✕</button>
        </div>
        {/* Body */}
        <div className="bg-white px-5 py-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            {[
              { label:"Event Type", val:typeLabel },
              { label:"Camera",     val:a.camera_id },
              { label:"Time",       val:a.ts },
              { label:"Status",     val:a.acked?"Acknowledged":"Active" },
            ].map(({ label, val }) => (
              <div key={label} className="bg-slate-50 rounded-lg px-3 py-2">
                <p className="text-[9px] text-slate-400 uppercase tracking-widest mb-0.5">{label}</p>
                <p className="text-xs font-semibold text-slate-800" style={{ fontFamily:"var(--font-mono)" }}>{val}</p>
              </div>
            ))}
          </div>
          {a.plate && (
            <div className="bg-slate-900 rounded-lg px-4 py-3 flex items-center justify-center gap-3">
              <span className="text-slate-400 text-[10px]">PLATE</span>
              <span className="text-white text-xl font-bold tracking-[0.25em]" style={{ fontFamily:"var(--font-mono)" }}>{a.plate}</span>
              {a.confidence && <span className="text-emerald-400 text-[10px]">{a.confidence}%</span>}
            </div>
          )}
          <div className="bg-slate-50 rounded-lg px-3 py-2">
            <p className="text-[9px] text-slate-400 uppercase tracking-widest mb-0.5">Details</p>
            <p className="text-xs text-slate-700">{a.message}. Camera {a.camera_id} flagged this event at {a.ts}. {a.type==="fence_crossing"?"Immediate response may be required.":a.type==="anpr"?"Cross-reference with MHA vehicle watchlist.":a.type==="face"?"Subject matched with confidence threshold >85%.":"Review feed and confirm alert."}</p>
          </div>
          <div className="flex gap-2 pt-1">
            {!a.acked && (
              <button onClick={()=>{onAck(a.id);onClose();}} className="flex-1 bg-slate-900 text-white text-xs font-semibold rounded-xl py-2.5 hover:bg-slate-700 transition-colors">
                Acknowledge
              </button>
            )}
            <button onClick={onClose} className={`${a.acked?"flex-1":""} bg-slate-100 text-slate-600 text-xs font-semibold rounded-xl py-2.5 px-4 hover:bg-slate-200 transition-colors`}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Alert item ───────────────────────────────────────────────────────────────

function AlertItem({ a, onAck }: { a: Alert; onAck: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const s = { critical:{bar:"bg-red-500",bg:"bg-red-50",border:"border-red-200"}, warning:{bar:"bg-amber-400",bg:"bg-amber-50",border:"border-amber-200"}, info:{bar:"bg-blue-400",bg:"bg-blue-50",border:"border-blue-200"} }[a.severity];
  const emoji = a.type==="anpr"?"🚗":a.type==="face"?"👤":a.type==="night"?"🌙":"⚠️";
  return (
    <>
      {open && <AlertModal a={a} onClose={()=>setOpen(false)} onAck={onAck} />}
      <div onClick={()=>setOpen(true)} className={`flex gap-3 rounded-xl border p-3 cursor-pointer hover:brightness-95 active:scale-[0.99] ${s.bg} ${s.border} ${a.acked?"opacity-40":"animate-fade-up"} transition-all`}>
        <div className={`w-0.5 self-stretch rounded-full flex-shrink-0 ${s.bar}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <span className="text-xs font-medium text-slate-800 leading-snug">{emoji} {a.message}</span>
            {!a.acked && <button onClick={e=>{e.stopPropagation();onAck(a.id);}} className="flex-shrink-0 text-[10px] rounded-md px-2 py-0.5 border border-current hover:bg-white/50 transition-colors" style={{ fontFamily:"var(--font-mono)", color:"inherit" }}>ACK</button>}
          </div>
          <div className="flex items-center gap-2 mt-1">
            {a.plate && <span className="bg-white border border-slate-200 rounded px-1.5 py-px text-[9px] tracking-widest" style={{ fontFamily:"var(--font-mono)" }}>{a.plate}</span>}
            <span className="text-[10px] text-slate-500" style={{ fontFamily:"var(--font-mono)" }}>{a.camera_id} · {a.ts}</span>
          </div>
        </div>
      </div>
    </>
  );
}

function SidebarCamRow({ cam, active, onClick }: { cam: CamConfig; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${active?"bg-white/10 border-r-2 border-blue-400":"hover:bg-white/6"}`}>
      <span className={`w-2 h-2 rounded-full flex-shrink-0 dot-pulse ${cam.online?(cam.isRecording?"bg-red-400":"bg-emerald-400"):"bg-slate-600"}`} />
      <div className="min-w-0">
        <p className="text-[13px] font-medium text-white/90 truncate leading-snug">{cam.label}</p>
        <p className="text-[10px] text-white/40 truncate" style={{ fontFamily:"var(--font-mono)" }}>{cam.id} · {cam.sector}{cam.isNight?" · NV":""}</p>
      </div>
    </button>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

type RightTab = "alerts" | "anpr" | "face" | "night";

export default function App() {
  const now = useNow();
  const [wsStatus, setWsStatus] = useState<"connected" | "demo">("demo");
  const [selectedCam, setSelectedCam] = useState("CAM-01");
  const [showBoxes, setShowBoxes] = useState(true);
  const snapshotRef = useRef<(() => void) | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("alerts");
  const [viewMode, setViewMode] = useState<"single" | "grid">("single");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Alerts
  const [alerts, setAlerts] = useState<Alert[]>([
    { id:"a1", type:"fence_crossing", severity:"critical", camera_id:"CAM-01", message:"Perimeter breach — North Gate Alpha", ts:"02:41:07", acked:false },
    { id:"a2", type:"anpr",           severity:"critical", camera_id:"CAM-03", message:"Watchlist vehicle: RJ14-CD-5678",    plate:"RJ14-CD-5678", confidence:97.4, ts:"02:39:52", acked:false },
    { id:"a3", type:"face",           severity:"warning",  camera_id:"CAM-01", message:"Watchlist subject — Ahmed Karimi",   ts:"02:38:14", acked:false },
    { id:"a4", type:"motion",         severity:"warning",  camera_id:"CAM-02", message:"Thermal anomaly — east flank",        ts:"02:35:00", acked:true  },
    { id:"a5", type:"night",          severity:"info",     camera_id:"CAM-04", message:"IR activated — lux below threshold",  ts:"02:30:45", acked:true  },
  ]);

  // ANPR
  const [anprScans, setAnprScans] = useState<ANPRScan[]>(() =>
    PLATE_POOL.slice(0, 5).map((p, i) => ({ ...p, id:`anpr-${i}`, ts:`02:${39-i*2}:${pad(Math.floor(Math.random()*59))}`, state:"done" as const }))
  );

  // Face
  const [faceScans, setFaceScans] = useState<FaceScan[]>(() =>
    FACE_POOL.slice(0, 4).map((f, i) => ({ ...f, id:`face-${i}`, ts:`02:${38-i*3}:${pad(Math.floor(Math.random()*59))}`, state:"done" as const }))
  );

  // Night
  const [nightData, setNightData] = useState<NightStatus[]>(
    CAMERAS.filter(c => c.online).map(c => ({ camera_id:c.id, label:c.label, irOn:c.isNight, lux:c.isNight?14+Math.random()*18:85+Math.random()*110, tempC:14+Math.random()*22, motionScore:Math.floor(Math.random()*75) }))
  );


  // ── Night sensor drift every 3s ────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      setNightData(prev => prev.map(n => ({
        ...n,
        lux: Math.max(0, n.lux + (Math.random()-0.5)*7),
        tempC: n.tempC + (Math.random()-0.5)*0.4,
        motionScore: clamp(n.motionScore + (Math.random()-0.42)*14, 0, 100),
      })));
    }, 3000);
    return () => clearInterval(id);
  }, []);


  // ── ANPR new scan every 7s ─────────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      const pick = PLATE_POOL[Math.floor(Math.random() * PLATE_POOL.length)];
      const scan: ANPRScan = { ...pick, id:`anpr-${Date.now()}`, ts:ts(now), state:"scanning" };
      setAnprScans(prev => [scan, ...prev].slice(0, 15));
      setTimeout(() => {
        setAnprScans(prev => prev.map(s => s.id === scan.id ? { ...s, state:"done" } : s));
        if (scan.status === "watchlist") {
          setAlerts(prev => [{ id:`al-anpr-${Date.now()}`, type:"anpr", severity:"critical", camera_id:scan.camera_id, message:`Watchlist plate: ${scan.plate}`, plate:scan.plate, confidence:scan.conf, ts:ts(now), acked:false }, ...prev].slice(0,40));
        }
      }, 2600);
    }, 7000);
    return () => clearInterval(id);
  }, [now]);

  // ── Face new scan every 9s ─────────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      const pick = FACE_POOL[Math.floor(Math.random() * FACE_POOL.length)];
      const scan: FaceScan = { ...pick, id:`face-${Date.now()}`, ts:ts(now), state:"scanning" };
      setFaceScans(prev => [scan, ...prev].slice(0, 12));
      setTimeout(() => {
        setFaceScans(prev => prev.map(s => s.id === scan.id ? { ...s, state:"done" } : s));
        if (scan.status === "watchlist") {
          setAlerts(prev => [{ id:`al-face-${Date.now()}`, type:"face", severity:"warning", camera_id:scan.camera_id, message:`Watchlist subject: ${scan.label}`, ts:ts(now), acked:false }, ...prev].slice(0,40));
        }
      }, 3100);
    }, 9000);
    return () => clearInterval(id);
  }, [now]);

  // WS attempt
  useEffect(() => {
    let ws: WebSocket;
    try {
      ws = new WebSocket("ws://localhost:8000/ws");
      ws.onopen  = () => setWsStatus("connected");
      ws.onclose = () => setWsStatus("demo");
      ws.onerror = () => setWsStatus("demo");
    } catch (_) { setWsStatus("demo"); }
    return () => { try { ws?.close(); } catch(_){} };
  }, []);

  const ackAlert = useCallback((id: string) => setAlerts(prev => prev.map(a => a.id===id ? {...a,acked:true} : a)), []);
  const toggleIR = useCallback((camId: string) => setNightData(prev => prev.map(n => n.camera_id===camId ? {...n,irOn:!n.irOn} : n)), []);

  const cam = CAMERAS.find(c => c.id === selectedCam)!;
  const unacked = alerts.filter(a => !a.acked).length;
  const critical = alerts.filter(a => !a.acked && a.severity==="critical").length;
  const nightActive = nightData.filter(n => n.irOn).length;
  const liveCams = CAMERAS.filter(c => c.online).length;

  const tabs: { key: RightTab; label: string; count?: number; icon: React.ReactNode }[] = [
    { key:"alerts", label:"Alerts",  count:unacked, icon:<Icon.Alert /> },
    { key:"anpr",   label:"ANPR",    count:anprScans.filter(s=>s.status==="watchlist").length, icon:<Icon.Plate /> },
    { key:"face",   label:"Face ID", count:faceScans.filter(s=>s.status==="watchlist").length, icon:<Icon.Face /> },
    { key:"night",  label:"Night",   count:nightActive, icon:<Icon.Moon /> },
  ];

  return (
    <div className="flex h-full overflow-hidden" style={{ fontFamily:"var(--font-sans)", background:"#f0f2f5" }}>

      {/* ── Sidebar ── */}
      {sidebarOpen && (
        <aside className="w-60 flex-shrink-0 flex flex-col overflow-hidden" style={{ background:"#0f172a" }}>
          <div className="px-5 pt-6 pb-5 border-b border-white/8">
            <div className="flex items-center gap-2.5 mb-1">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0 text-white"><Icon.Camera /></div>
              <div><p className="text-[11px] font-bold text-white uppercase tracking-widest leading-none">IBVAP</p><p className="text-[9px] text-white/40 uppercase tracking-widest mt-0.5">SIH · 26187</p></div>
            </div>
          </div>
          <div className="px-5 py-4 border-b border-white/8">
            <p className="text-[11px] text-white/40" style={{ fontFamily:"var(--font-mono)" }}>{pad(now.getHours())}:{pad(now.getMinutes())}:{pad(now.getSeconds())} IST</p>
            <p className="text-base font-semibold text-white mt-0.5">{greeting(now.getHours())}, Officer</p>
            <div className="flex items-center gap-1.5 mt-2">
              <Dot color={wsStatus==="connected"?"bg-emerald-400":"bg-amber-400"} pulse />
              <span className="text-[10px] text-white/50" style={{ fontFamily:"var(--font-mono)" }}>{wsStatus==="connected"?"Live · Backend":"Demo Mode"}</span>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto py-2">
            <p className="px-5 py-2 text-[9px] font-semibold text-white/30 uppercase tracking-widest">Feeds ({liveCams}/{CAMERAS.length})</p>
            {CAMERAS.map(c => <SidebarCamRow key={c.id} cam={c} active={selectedCam===c.id} onClick={()=>setSelectedCam(c.id)} />)}
          </div>
          <div className="border-t border-white/8 px-5 py-4">
            <p className="text-[9px] font-semibold text-white/30 uppercase tracking-widest mb-2">Device Info</p>
            {[{k:"Hikvision DS-2CD",v:"CAM-01"},{k:"Dahua IPC-HDW",v:"CAM-02"},{k:"Axis P3245-V",v:"CAM-03"}].map(({k,v})=>(
              <div key={k} className="flex justify-between py-0.5"><span className="text-[10px] text-white/55 truncate">{k}</span><span className="text-[9px] text-white/30" style={{ fontFamily:"var(--font-mono)" }}>{v}</span></div>
            ))}
          </div>
        </aside>
      )}

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* ── Top bar ── */}
        <header className="flex items-center justify-between px-5 h-14 bg-white border-b border-slate-200 flex-shrink-0">
          <div className="flex items-center gap-3">
            <button onClick={()=>setSidebarOpen(v=>!v)} className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect y="2.5" width="16" height="1.5" rx=".75" fill="currentColor"/><rect y="7.25" width="16" height="1.5" rx=".75" fill="currentColor"/><rect y="12" width="16" height="1.5" rx=".75" fill="currentColor"/></svg>
            </button>
            <div>
              <p className="text-sm font-semibold text-slate-800">{cam.label}</p>
              <p className="text-[10px] text-slate-400" style={{ fontFamily:"var(--font-mono)" }}>{cam.coords} · Kashmir Border · {cam.sector}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {unacked > 0 && (
              <button onClick={()=>setRightTab("alerts")} className="relative flex items-center gap-1.5 bg-red-50 border border-red-200 rounded-lg px-3 py-1.5 text-red-600 hover:bg-red-100 transition-colors">
                <Icon.Alert /><span className="text-xs font-semibold">{unacked} Alert{unacked>1?"s":""}</span>
                {critical>0 && <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-white flex items-center justify-center font-bold" style={{ fontSize:9 }}>{critical}</span>}
              </button>
            )}

            {/* ── DETECT BOXES TOGGLE ── */}
            <button
              onClick={() => setShowBoxes(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                showBoxes
                  ? "bg-blue-600 text-white border-blue-700 shadow-sm shadow-blue-200"
                  : "bg-white text-slate-500 border-slate-200 hover:border-slate-300"
              }`}
            >
              <Icon.Box />
              {showBoxes ? "Boxes ON" : "Boxes OFF"}
            </button>

            <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-1">
              <button onClick={()=>setViewMode("single")} className={`px-2 py-1 rounded-md text-xs font-medium transition-all ${viewMode==="single"?"bg-white shadow-sm text-slate-800":"text-slate-500 hover:text-slate-700"}`}>Single</button>
              <button onClick={()=>setViewMode("grid")} className={`px-2 py-1 rounded-md text-xs font-medium transition-all flex items-center gap-1 ${viewMode==="grid"?"bg-white shadow-sm text-slate-800":"text-slate-500 hover:text-slate-700"}`}><Icon.Grid /> Grid</button>
            </div>

            <div className="flex items-center gap-1">
              <button onClick={()=>snapshotRef.current?.()} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-600 hover:bg-slate-100 border border-slate-200 transition-colors"><Icon.Snapshot /> Snapshot</button>
              <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-600 text-white hover:bg-red-700 transition-colors"><span className="w-1.5 h-1.5 rounded-full bg-white dot-pulse" /> REC</button>
            </div>
          </div>
        </header>

        {/* ── Stat strip ── */}
        <div className="flex items-center gap-2 px-5 py-3 bg-white border-b border-slate-100 flex-shrink-0 overflow-x-auto">
          {[
            { icon:<Icon.Alert />, label:"Unacked Alerts",  val:unacked,          col:unacked>0?"text-red-600":"text-slate-700",  extra:unacked>0?"bg-red-50 border-red-200":"" },
            { icon:<Icon.Plate />, label:"ANPR Scans",       val:anprScans.length, col:"text-slate-700" },
            { icon:<Icon.Face />,  label:"Face Scans",       val:faceScans.length, col:"text-blue-600" },
            { icon:<Icon.Alert />, label:"Watchlist Hits",   val:anprScans.filter(s=>s.status==="watchlist").length+faceScans.filter(s=>s.status==="watchlist").length, col:"text-amber-600", extra:"bg-amber-50 border-amber-200" },
            { icon:<Icon.Moon />,  label:"IR Active",         val:nightActive,      col:"text-emerald-600" },
            { icon:<Icon.Box />,   label:"AI Detection",  val:"ON", col:"text-blue-600" },
          ].map(({ icon, label, val, col, extra }) => (
            <div key={label} className={`flex items-center gap-2 bg-white border border-slate-200 rounded-xl px-4 py-2 flex-shrink-0 ${extra??""}`}>
              <span className={col}>{icon}</span>
              <div><p className={`text-lg font-bold leading-none ${col}`}>{val}</p><p className="text-[10px] text-slate-400 mt-0.5">{label}</p></div>
            </div>
          ))}
        </div>

        {/* ── Feed + Right ── */}
        <div className="flex flex-1 min-h-0 overflow-hidden">

          {/* Feed */}
          <div className="flex-1 p-4 overflow-y-auto min-w-0">
            {viewMode === "single" ? (
              <>
                <CameraFeed cam={cam} selected isSelected={true} onClick={()=>{}} compact={false} showBoxes={showBoxes} snapshotRef={snapshotRef} />
                <div className="mt-3 grid grid-cols-5 gap-2">
                  {CAMERAS.filter(c=>c.id!==selectedCam).map(c=>(
                    <div key={c.id}>
                      <CameraFeed cam={c} selected={false} isSelected={false} onClick={()=>setSelectedCam(c.id)} compact showBoxes={showBoxes} />
                      <p className="mt-1 text-[10px] text-slate-500 truncate">{c.label}</p>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                {CAMERAS.map(c=>(
                  <div key={c.id}>
                    <CameraFeed cam={c} selected={c.id===selectedCam} isSelected={c.id===selectedCam} onClick={()=>{setSelectedCam(c.id);setViewMode("single");}} compact={false} showBoxes={showBoxes} />
                    <p className="mt-1.5 text-[11px] font-medium text-slate-600">{c.label}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Right panel */}
          <aside className="w-80 flex-shrink-0 flex flex-col border-l border-slate-200 bg-white overflow-hidden">
            <div className="flex border-b border-slate-200 flex-shrink-0">
              {tabs.map(tab=>(
                <button key={tab.key} onClick={()=>setRightTab(tab.key)} className={`flex-1 flex items-center justify-center gap-1 py-3 text-[10px] font-medium transition-colors border-b-2 ${rightTab===tab.key?"border-blue-500 text-blue-600":"border-transparent text-slate-400 hover:text-slate-700"}`}>
                  <span className="opacity-70">{tab.icon}</span>{tab.label}
                  {tab.count!=null&&tab.count>0&&<span className={`w-4 h-4 rounded-full text-[9px] flex items-center justify-center font-bold ${rightTab===tab.key?"bg-blue-600 text-white":"bg-slate-200 text-slate-500"}`}>{tab.count}</span>}
                </button>
              ))}
            </div>
            <div className="flex-1 overflow-hidden flex flex-col">
              {rightTab==="alerts" && <div className="flex-1 overflow-y-auto p-3 space-y-2">{alerts.map(a=><AlertItem key={a.id} a={a} onAck={ackAlert} />)}</div>}
              {rightTab==="anpr"   && <ANPRPanel scans={anprScans} />}
              {rightTab==="face"   && <FacePanel scans={faceScans} />}
              {rightTab==="night"  && <NightPanel nightData={nightData} onToggle={toggleIR} />}
            </div>
            <div className="border-t border-slate-200 px-4 py-2.5 flex items-center justify-between flex-shrink-0">
              <div className="flex items-center gap-1.5">
                <Dot color={wsStatus==="connected"?"bg-emerald-400":"bg-amber-400"} pulse />
                <span className="text-[10px] text-slate-500" style={{ fontFamily:"var(--font-mono)" }}>{wsStatus==="connected"?"WS · localhost:8000":"Demo · No backend"}</span>
              </div>
              <span className="text-[10px] text-slate-400" style={{ fontFamily:"var(--font-mono)" }}>v2.4.1</span>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
