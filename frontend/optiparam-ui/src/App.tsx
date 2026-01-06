import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

type View = "Prediction" | "Recommendation";

interface Health {
  models_loaded?: number;
  version?: string;
}

interface SpecsPoint {
  lo: number;
  hi: number;
  target?: number;
  w?: number;
}

type SpecsMap = Record<string, SpecsPoint>;

interface BoundEntry {
  mid?: number;
  lo?: number;
  hi?: number;
  [key: string]: any;
}

type BoundsMap = Record<string, BoundEntry>;

interface GunRow {
  id: number;
  Rec: string;
  Gun: string;
  Powder: number;
  Air: number;
  kV: number;
  "µA": number;
  EClean: number;
  OnOff: number;
}

interface AxisRow {
  id: number;
  Rec: string;
  Axis: string;
  Upper: number;
  Lower: number;
  Speed: number;
  SprayDist: number;
  OnOff: number;
}

interface PredictionResponse {
  predictions: Record<string, number>;
  in_spec: Record<string, boolean>;
  margins: Record<string, number>;
  model_version?: string;
}

interface RecommendationResponse {
  recommended: Record<string, number>;
  predicted: Record<string, number>;
  model_version?: string;
}

interface DeltaRow {
  Feature: string;
  Current: number;
  Proposed: number;
  Delta: number;
}

interface TrialStartResponse {
  trial_code: string;
}

// how long after a trial action before we re-show the modal (ms)
const TRIAL_RESET_DELAY_MS = 60_000; // 1 minute – adjust as you like

// ---------- Helper functions (ported from your Python) ----------

function mid(bounds: BoundsMap, name: string, defaultValue = 0): number {
  const b = bounds[name];
  if (!b || b.mid === undefined || b.mid === null) return defaultValue;
  return Number(b.mid);
}

const gunLayout: Array<{ Rec: string; Gun: string }> = [
  { Rec: "A", Gun: "1" },
  { Rec: "B", Gun: "1" },
  { Rec: "C", Gun: "1" },
  { Rec: "C", Gun: "2" },
  { Rec: "D", Gun: "1" },
  { Rec: "D", Gun: "2" },
  { Rec: "E", Gun: "1" },
  { Rec: "F", Gun: "1" },
];

function gunField(rec: string, gun: string, colKey: string): string {
  const suffixMap: Record<string, string> = {
    Powder: "Powder",
    Air: "Total_Air",
    kV: "Voltage",
    "µA": "Current",
    EClean: "Electrode_Cleaning",
    OnOff: "onoff_before_after_obj",
  };
  const suffix = suffixMap[colKey];
  return `Rec_${rec}_Gun${gun}_${suffix}`;
}

function axisField(rec: string, axis: string, colKey: string): string {
  if (colKey === "OnOff") {
    if (rec === "D" && axis === "Z") {
      return "Rec_D_Axis-Z_Spray_onoff_before_after";
    }
    return `Rec_${rec}_Axis-${axis}_onoff_before_after`;
  }
  const suffixMap: Record<string, string> = {
    Upper: "Upper_Point",
    Lower: "Lower_Point",
    Speed: "Speed",
    SprayDist: "Spray_Dist",
  };
  const suffix = suffixMap[colKey];
  return `Rec_${rec}_Axis-${axis}_${suffix}`;
}

function defaultGuns(bounds: BoundsMap): GunRow[] {
  return gunLayout.map((g, idx) => ({
    id: idx,
    Rec: g.Rec,
    Gun: g.Gun,
    Powder: mid(bounds, gunField(g.Rec, g.Gun, "Powder"), 0),
    Air: mid(bounds, gunField(g.Rec, g.Gun, "Air"), 0),
    kV: mid(bounds, gunField(g.Rec, g.Gun, "kV"), 0),
    "µA": mid(bounds, gunField(g.Rec, g.Gun, "µA"), 0),
    EClean: mid(bounds, gunField(g.Rec, g.Gun, "EClean"), 0),
    OnOff: mid(bounds, gunField(g.Rec, g.Gun, "OnOff"), 0),
  }));
}

function defaultAxes(bounds: BoundsMap): AxisRow[] {
  const rows: AxisRow[] = [];
  let id = 0;
  for (const rec of ["A", "B", "C", "D", "E", "F"]) {
    for (const axis of ["Z", "X"]) {
      rows.push({
        id: id++,
        Rec: rec,
        Axis: axis,
        Upper: mid(bounds, axisField(rec, axis, "Upper"), 0),
        Lower: mid(bounds, axisField(rec, axis, "Lower"), 0),
        Speed: mid(bounds, axisField(rec, axis, "Speed"), 0),
        SprayDist: mid(bounds, axisField(rec, axis, "SprayDist"), 0),
        OnOff: mid(bounds, axisField(rec, axis, "OnOff"), 0),
      });
    }
  }
  return rows;
}

function gunsRowsToFeatures(rows: GunRow[]): Record<string, number> {
  const features: Record<string, number> = {};
  rows.forEach((r) => {
    const rec = String(r.Rec);
    const gun = String(r.Gun);
    ["Powder", "Air", "kV", "µA", "EClean", "OnOff"].forEach((k) => {
      const v = Number((r as any)[k] ?? 0);
      features[gunField(rec, gun, k)] = isNaN(v) ? 0 : v;
    });
  });
  return features;
}

function axesRowsToFeatures(rows: AxisRow[]): Record<string, number> {
  const features: Record<string, number> = {};
  rows.forEach((r) => {
    const rec = String(r.Rec);
    const axis = String(r.Axis);
    ["Upper", "Lower", "Speed", "SprayDist", "OnOff"].forEach((k) => {
      const v = Number((r as any)[k] ?? 0);
      features[axisField(rec, axis, k)] = isNaN(v) ? 0 : v;
    });
  });
  return features;
}

function applyToGuns(base: GunRow[], features: Record<string, number>): GunRow[] {
  return base.map((row) => {
    const newRow: GunRow = { ...row };
    ["Powder", "Air", "kV", "µA", "EClean", "OnOff"].forEach((k) => {
      const fname = gunField(row.Rec, row.Gun, k);
      if (features.hasOwnProperty(fname)) {
        (newRow as any)[k] = features[fname];
      }
    });
    return newRow;
  });
}

function applyToAxes(base: AxisRow[], features: Record<string, number>): AxisRow[] {
  return base.map((row) => {
    const newRow: AxisRow = { ...row };
    ["Upper", "Lower", "Speed", "SprayDist", "OnOff"].forEach((k) => {
      const fname = axisField(row.Rec, row.Axis, k);
      if (features.hasOwnProperty(fname)) {
        (newRow as any)[k] = features[fname];
      }
    });
    return newRow;
  });
}

function buildDeltas(
  current: Record<string, number>,
  proposed: Record<string, number>
): DeltaRow[] {
  const rows: DeltaRow[] = [];
  Object.entries(proposed).forEach(([k, v]) => {
    const cur = current[k];
    if (cur === undefined || cur === null) return;
    const curNum = Number(cur);
    const newNum = Number(v);
    if (isNaN(curNum) || isNaN(newNum)) return;
    rows.push({
      Feature: k,
      Current: curNum,
      Proposed: newNum,
      Delta: newNum - curNum,
    });
  });
  return rows.sort((a, b) => a.Feature.localeCompare(b.Feature));
}

// ---------- Small helpers ----------

async function apiGet<T>(base: string, path: string): Promise<T> {
  const resp = await fetch(`${base}${path}`);
  if (!resp.ok) {
    throw new Error(`GET ${path} ${resp.status}`);
  }
  return resp.json();
}

async function apiPost<T>(
  base: string,
  path: string,
  payload: any
): Promise<{ data: T; resp: Response }> {
  const resp = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`POST ${path} ${resp.status}: ${text}`);
  }
  const data = (await resp.json()) as T;
  return { data, resp };
}

function nearlyEqual(a: number, b: number, eps = 1e-9): boolean {
  return Math.abs(a - b) <= eps;
}

// ---------- Main component ----------

const App: React.FC = () => {
  const [apiBase, setApiBase] = useState("http://10.175.24.242:8000");
  const [health, setHealth] = useState<Health | null>(null);
  const [types, setTypes] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState<string>("");

  const [specs, setSpecs] = useState<SpecsMap>({});
  const [bounds, setBounds] = useState<BoundsMap>({});

  const [view, setView] = useState<View>("Prediction");

  const [humidity, setHumidity] = useState(50);
  const [temperature, setTemperature] = useState(23);
  const [valveStatus, setValveStatus] = useState(0);

  const [gunsRows, setGunsRows] = useState<GunRow[]>([]);
  const [axesRows, setAxesRows] = useState<AxisRow[]>([]);

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predMeta, setPredMeta] = useState<{ reqId?: string | null; runId?: string | null }>(
    {}
  );

  const [recommendation, setRecommendation] =
    useState<RecommendationResponse | null>(null);
  const [recMeta, setRecMeta] = useState<{ reqId?: string | null; runId?: string | null }>(
    {}
  );

  // Snapshot of "current" at the time we clicked Recommend (freeze highlight baseline)
  const [snapshotGunsRows, setSnapshotGunsRows] = useState<GunRow[]>([]);
  const [snapshotAxesRows, setSnapshotAxesRows] = useState<AxisRow[]>([]);
  const [lastCurrentFeatures, setLastCurrentFeatures] = useState<Record<string, number>>({});

  // Recommendation form state
  const defaultTargets = useMemo(() => {
    const t: Record<string, number> = {};
    Object.entries(specs || {}).forEach(([p, s]) => {
      t[p] = (s.lo + s.hi) / 2.0;
    });
    return t;
  }, [specs]);

  const [targets, setTargets] = useState<Record<string, number>>({});
  const [ctxHumidity, setCtxHumidity] = useState(50);
  const [ctxTemp, setCtxTemp] = useState(23);
  const [ctxValve, setCtxValve] = useState(0);

  const [trials, setTrials] = useState(150);
  const [timeoutSec, setTimeoutSec] = useState(15);
  const [maxStepPct, setMaxStepPct] = useState(2.0);

  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingRec, setLoadingRec] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 🔐 Trial state
  const [trialCode, setTrialCode] = useState<string | null>(null);
  const [trialStarted, setTrialStarted] = useState(false);
  const [showTrialModal, setShowTrialModal] = useState(true);
  const [trialStage, setTrialStage] = useState<"choices" | "code">("choices");
  const [isStartingTrial, setIsStartingTrial] = useState(false);

  // timer ref so we can auto-reopen modal after a trial action
  const trialTimeoutRef = useRef<number | null>(null);

  const clearTrialTimeout = () => {
    if (trialTimeoutRef.current !== null) {
      window.clearTimeout(trialTimeoutRef.current);
      trialTimeoutRef.current = null;
    }
  };

  const scheduleTrialReset = () => {
    clearTrialTimeout();
    trialTimeoutRef.current = window.setTimeout(() => {
      setShowTrialModal(true);
      setTrialStage("choices");
      setTrialStarted(false);
      setTrialCode(null);
      trialTimeoutRef.current = null;
    }, TRIAL_RESET_DELAY_MS);
  };

  useEffect(() => {
    // clear timeout on unmount
    return () => {
      clearTrialTimeout();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ----- Sync view with ?view=... in URL -----
  useEffect(() => {
    const url = new URL(window.location.href);
    const v = url.searchParams.get("view");
    if (v === "Prediction" || v === "Recommendation") {
      setView(v);
    }
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("view", view);
    window.history.replaceState({}, "", url.toString());
  }, [view]);

  // ----- Load health + types -----
  useEffect(() => {
    const loadBase = async () => {
      try {
        setError(null);
        const h = await apiGet<Health>(apiBase, "/health");
        setHealth(h);
        const tResp = await apiGet<{ types: string[] }>(apiBase, "/types");
        const ts = tResp.types || [];
        setTypes(ts);
        if (ts.length > 0 && !selectedType) {
          setSelectedType(ts[0]);
        }
      } catch (err: any) {
        console.error(err);
        setError(err.message || String(err));
        setHealth(null);
        setTypes([]);
      }
    };
    loadBase();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  // ----- Load specs + bounds when type changes -----
  useEffect(() => {
    if (!selectedType) return;
    const loadTypeData = async () => {
      try {
        setError(null);
        const specsResp = await apiGet<{ specs: SpecsMap }>(apiBase, `/specs/${selectedType}`);
        const bResp = await apiGet<{ bounds: BoundsMap }>(apiBase, `/feature-bounds/${selectedType}`);

        setSpecs(specsResp.specs || {});
        setBounds(bResp.bounds || {});

        const b = bResp.bounds || {};
        const dg = defaultGuns(b);
        const da = defaultAxes(b);

        setGunsRows(dg);
        setAxesRows(da);

        setHumidity(b.Humidity?.mid ?? 50);
        setTemperature(b.Temperature?.mid ?? 23);
        setValveStatus(b.Valve_Filter_Status?.mid ?? 0);

        setCtxHumidity(b.Humidity?.mid ?? 50);
        setCtxTemp(b.Temperature?.mid ?? 23);
        setCtxValve(b.Valve_Filter_Status?.mid ?? 0);

        setPrediction(null);
        setRecommendation(null);

        // reset snapshots + highlight baseline
        setSnapshotGunsRows([]);
        setSnapshotAxesRows([]);
        setLastCurrentFeatures({});

        setTargets({});
      } catch (err: any) {
        console.error(err);
        setError(err.message || String(err));
      }
    };
    loadTypeData();
  }, [apiBase, selectedType]);

  // reset targets when specs change
  useEffect(() => {
    if (Object.keys(defaultTargets).length > 0) {
      setTargets(defaultTargets);
    }
  }, [defaultTargets]);

  // ---------- Small handlers ----------

  const statusHtml = health
    ? `模型版本：${health.version ?? "—"} • 已加载模型：${health.models_loaded ?? "—"}`
    : "API 不可用";

  const handleGunChange = (rowId: number, key: keyof GunRow, value: number) => {
    setGunsRows((rows) => rows.map((r) => (r.id === rowId ? { ...r, [key]: value } : r)));
  };

  const handleAxisChange = (rowId: number, key: keyof AxisRow, value: number) => {
    setAxesRows((rows) => rows.map((r) => (r.id === rowId ? { ...r, [key]: value } : r)));
  };

  const handleResetMedians = () => {
    const b = bounds || {};
    setGunsRows(defaultGuns(b));
    setAxesRows(defaultAxes(b));

    // Clear any previous recommendation highlighting baseline
    setSnapshotGunsRows([]);
    setSnapshotAxesRows([]);
    setLastCurrentFeatures({});

    setPrediction(null);
    setRecommendation(null);
  };

  const handlePredict = async () => {
    if (!selectedType) return;

    if (!trialStarted || !trialCode) {
      alert("请先点击“开始试运行”，获得试运行编号后再进行“预测”。");
      return;
    }

    try {
      setError(null);
      setLoadingPredict(true);

      const feat: Record<string, number> = {
        ...gunsRowsToFeatures(gunsRows),
        ...axesRowsToFeatures(axesRows),
        Humidity: humidity,
        Temperature: temperature,
        Valve_Filter_Status: valveStatus,
      };

      const payload = {
        type_code: selectedType,
        features: feat,
        trial_code: trialCode,
      };

      const { data, resp } = await apiPost<PredictionResponse>(apiBase, "/predict", payload);

      setPrediction(data);
      setPredMeta({
        reqId: resp.headers.get("X-Request-ID"),
        runId: resp.headers.get("X-Run-ID"),
      });
      setView("Prediction");

      // schedule modal to reappear so they can start a new trial
      scheduleTrialReset();
    } catch (err: any) {
      console.error(err);
      setError(err.message || String(err));
    } finally {
      setLoadingPredict(false);
    }
  };

  const handleRecommend = async () => {
    if (!selectedType) return;

    if (!trialStarted || !trialCode) {
      alert("请先点击“开始试运行”，获得试运行编号后再进行“推荐”。");
      return;
    }

    try {
      setError(null);
      setLoadingRec(true);

      // Freeze "current" snapshot for highlighting + stable display
      const frozenGuns = gunsRows.map((r) => ({ ...r }));
      const frozenAxes = axesRows.map((r) => ({ ...r }));
      setSnapshotGunsRows(frozenGuns);
      setSnapshotAxesRows(frozenAxes);

      const current = {
        ...gunsRowsToFeatures(frozenGuns),
        ...axesRowsToFeatures(frozenAxes),
      };

      const payload = {
        type_code: selectedType,
        targets,
        fixed_context: {
          Humidity: ctxHumidity,
          Temperature: ctxTemp,
          Valve_Filter_Status: ctxValve,
        },
        current,
        step_pct: maxStepPct / 100.0,
        n_trials: trials,
        timeout_sec: timeoutSec,
        trial_code: trialCode,
      };

      const { data, resp } = await apiPost<RecommendationResponse>(apiBase, "/recommend", payload);

      setRecommendation(data);
      setRecMeta({
        reqId: resp.headers.get("X-Request-ID"),
        runId: resp.headers.get("X-Run-ID"),
      });

      // This is the baseline used for highlighting changes
      setLastCurrentFeatures(current);

      setView("Recommendation");

      // schedule modal to reappear so they can start a new trial
      scheduleTrialReset();
    } catch (err: any) {
      console.error(err);
      setError(err.message || String(err));
    } finally {
      setLoadingRec(false);
    }
  };

  // Build recommended tables from the FROZEN snapshot (not the live editor)
  const recGunsRows = useMemo(() => {
    if (!recommendation) return [];
    const base = snapshotGunsRows.length > 0 ? snapshotGunsRows : gunsRows;
    return applyToGuns(base, recommendation.recommended || {});
  }, [recommendation, snapshotGunsRows, gunsRows]);

  const recAxesRows = useMemo(() => {
    if (!recommendation) return [];
    const base = snapshotAxesRows.length > 0 ? snapshotAxesRows : axesRows;
    return applyToAxes(base, recommendation.recommended || {});
  }, [recommendation, snapshotAxesRows, axesRows]);

  const deltaRows: DeltaRow[] = useMemo(
    () => (recommendation ? buildDeltas(lastCurrentFeatures, recommendation.recommended || {}) : []),
    [recommendation, lastCurrentFeatures]
  );

  const predRowsFromPrediction = useMemo(() => {
    if (!prediction) return [];
    const rows: Array<{ Point: string; Predicted: number; InSpec: boolean; Margin: number }> = [];

    Object.entries(prediction.predictions || {}).forEach(([k, v]) => {
      const spec = specs[k] || { lo: -1e9, hi: 1e9 };
      const inSpec = v >= spec.lo && v <= spec.hi;
      const margin = Math.min(v - spec.lo, spec.hi - v);
      rows.push({ Point: k, Predicted: v, InSpec: inSpec, Margin: margin });
    });

    rows.sort((a, b) => a.Point.localeCompare(b.Point));
    return rows;
  }, [prediction, specs]);

  const predRowsFromRecommendation = useMemo(() => {
    if (!recommendation) return [];
    const rows: Array<{ Point: string; Predicted: number; InSpec: boolean; Margin: number }> = [];

    Object.entries(recommendation.predicted || {}).forEach(([k, v]) => {
      const spec = specs[k] || { lo: -1e9, hi: 1e9 };
      const inSpec = v >= spec.lo && v <= spec.hi;
      const margin = Math.min(v - spec.lo, spec.hi - v);
      rows.push({ Point: k, Predicted: v, InSpec: inSpec, Margin: margin });
    });

    rows.sort((a, b) => a.Point.localeCompare(b.Point));
    return rows;
  }, [recommendation, specs]);

  const handleApplyRecommended = () => {
    setGunsRows(recGunsRows);
    setAxesRows(recAxesRows);
    setView("Prediction");
  };

  const handleDownloadJSON = () => {
    if (!recommendation) return;
    const blob = new Blob([JSON.stringify(recommendation, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "recommendation.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearRecommendation = () => {
    setRecommendation(null);
    setSnapshotGunsRows([]);
    setSnapshotAxesRows([]);
    setLastCurrentFeatures({});
  };

  const startTrialFlow = async () => {
    try {
      clearTrialTimeout();
      setIsStartingTrial(true);
      const resp = await fetch(`${apiBase}/trial/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Failed to start trial: ${resp.status} ${text}`);
      }
      const data: TrialStartResponse = await resp.json();
      setTrialCode(data.trial_code);
      setTrialStarted(true);
      setTrialStage("code");
    } catch (err) {
      console.error(err);
      alert("开始试运行失败，请检查 API 后重试。");
    } finally {
      setIsStartingTrial(false);
    }
  };

  // ---------- Render helpers ----------

  const renderSpecsTable = () => {
    if (!specs || Object.keys(specs).length === 0) {
      return (
        <div className="alert alert-error">
          未找到厚度规格数据。请先写入数据库或运行训练以生成 dbo.thickness_spec。
        </div>
      );
    }

    const rows = Object.entries(specs)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([p, s]) => ({
        Point: p,
        Lower: s.lo,
        Target: s.target,
        Upper: s.hi,
        Weight: s.w ?? 1.0,
      }));

    return (
      <table className="table">
        <thead>
          <tr>
            <th>点位</th>
            <th>下限 (μm)</th>
            <th>目标 (μm)</th>
            <th>上限 (μm)</th>
            <th>权重</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.Point}>
              <td>{r.Point}</td>
              <td>{r.Lower}</td>
              <td>{r.Target}</td>
              <td>{r.Upper}</td>
              <td>{r.Weight}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const renderGunsTable = (
    rows: GunRow[],
    editable: boolean,
    highlightAgainst?: Record<string, number>
  ) => {
    const columns: Array<{ key: keyof GunRow; label: string }> = [
      { key: "Rec", label: "Rec" },
      { key: "Gun", label: "枪号" },
      { key: "Powder", label: "粉末" },
      { key: "Air", label: "总风量" },
      { key: "kV", label: "电压(kV)" },
      { key: "µA", label: "电流(µA)" },
      { key: "EClean", label: "电极清洁" },
      { key: "OnOff", label: "开/关" },
    ];

    return (
      <table className="table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={String(c.key)}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              {columns.map((c) => {
                const key = c.key;
                const value = row[key] as number | string;
                const isEditable = editable && !["Rec", "Gun"].includes(key as string);

                if (isEditable) {
                  return (
                    <td key={String(key)}>
                      <input
                        type="number"
                        className="cell-input"
                        value={value}
                        onChange={(e) => handleGunChange(row.id, key, Number(e.target.value))}
                      />
                    </td>
                  );
                }

                // Non-editable: optionally highlight if recommended != current snapshot
                let cell: React.ReactNode = value;
                const isDataCell = !["Rec", "Gun"].includes(key as string);

                if (highlightAgainst && isDataCell) {
                  const fname = gunField(String(row.Rec), String(row.Gun), String(key));
                  const cur = highlightAgainst[fname];
                  const next = Number(value);
                  if (cur !== undefined && !isNaN(next) && !nearlyEqual(Number(cur), next)) {
                    cell = <span className="cell-highlight">{value}</span>;
                  }
                }

                return <td key={String(key)}>{cell}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const renderAxesTable = (
    rows: AxisRow[],
    editable: boolean,
    highlightAgainst?: Record<string, number>
  ) => {
    const columns: Array<{ key: keyof AxisRow; label: string }> = [
      { key: "Rec", label: "Rec" },
      { key: "Axis", label: "轴" },
      { key: "Upper", label: "上点位" },
      { key: "Lower", label: "下点位" },
      { key: "Speed", label: "速度" },
      { key: "SprayDist", label: "喷涂距离" },
      { key: "OnOff", label: "开/关" },
    ];

    return (
      <table className="table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={String(c.key)}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              {columns.map((c) => {
                const key = c.key;
                const value = row[key] as number | string;
                const isEditable = editable && !["Rec", "Axis"].includes(key as string);

                if (isEditable) {
                  return (
                    <td key={String(key)}>
                      <input
                        type="number"
                        className="cell-input"
                        value={value}
                        onChange={(e) => handleAxisChange(row.id, key, Number(e.target.value))}
                      />
                    </td>
                  );
                }

                // Non-editable: optionally highlight if recommended != current snapshot
                let cell: React.ReactNode = value;
                const isDataCell = !["Rec", "Axis"].includes(key as string);

                if (highlightAgainst && isDataCell) {
                  const fname = axisField(String(row.Rec), String(row.Axis), String(key));
                  const cur = highlightAgainst[fname];
                  const next = Number(value);
                  if (cur !== undefined && !isNaN(next) && !nearlyEqual(Number(cur), next)) {
                    cell = <span className="cell-highlight">{value}</span>;
                  }
                }

                return <td key={String(key)}>{cell}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const renderPredictionResults = () => {
    if (!prediction) return null;
    return (
      <div className="card">
        <div className="card-title">预测结果</div>
        <table className="table">
          <thead>
            <tr>
              <th>点位</th>
              <th>预测值 (μm)</th>
              <th>是否合格</th>
              <th>离边界裕量 (μm)</th>
            </tr>
          </thead>
          <tbody>
            {predRowsFromPrediction.map((r) => (
              <tr key={r.Point}>
                <td>{r.Point}</td>
                <td>{r.Predicted.toFixed(2)}</td>
                <td>{r.InSpec ? "✅" : "❌"}</td>
                <td>{r.Margin.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="caption">
          模型版本：{prediction.model_version ?? "—"} • 请求：{predMeta.reqId ?? "—"} • 运行：{" "}
          {predMeta.runId ?? "—"}
        </div>
      </div>
    );
  };

  const renderRecommendationSection = () => {
    if (!recommendation) return null;

    return (
      <>
        <div className="card">
          <div className="card-title">推荐参数 – 喷枪</div>
          <div className="caption">绿色高亮 = 与当前参数相比，推荐后需要调整的项</div>
          {renderGunsTable(recGunsRows, false, lastCurrentFeatures)}
        </div>

        <div className="card">
          <div className="card-title">推荐参数 – 轴参数</div>
          <div className="caption">绿色高亮 = 与当前参数相比，推荐后需要调整的项</div>
          {renderAxesTable(recAxesRows, false, lastCurrentFeatures)}
        </div>

        {/* Delta table kept disabled for now */}
        {false && (
          <div className="card">
            <div className="card-title">建议调整项（可调特征）</div>
            {deltaRows.length > 0 ? (
              <table className="table">
                <thead>
                  <tr>
                    <th>特征</th>
                    <th>当前值</th>
                    <th>建议值</th>
                    <th>变化量</th>
                  </tr>
                </thead>
                <tbody>
                  {deltaRows.map((r) => (
                    <tr key={r.Feature}>
                      <td>{r.Feature}</td>
                      <td>{r.Current.toFixed(2)}</td>
                      <td>{r.Proposed.toFixed(2)}</td>
                      <td>{r.Delta.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="caption">没有给出可调特征的变化建议（可能当前设置已接近最优）。</div>
            )}
          </div>
        )}

        <div className="card">
          <div className="card-title">预测厚度 (μm)</div>
          <table className="table">
            <thead>
              <tr>
                <th>点位</th>
                <th>预测值 (μm)</th>
                <th>是否合格</th>
                <th>离边界裕量 (μm)</th>
              </tr>
            </thead>
            <tbody>
              {predRowsFromRecommendation.map((r) => (
                <tr key={r.Point}>
                  <td>{r.Point}</td>
                  <td>{r.Predicted.toFixed(2)}</td>
                  <td>{r.InSpec ? "✅" : "❌"}</td>
                  <td>{r.Margin.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="caption">
            模型版本：{recommendation.model_version ?? "—"} • 请求：{recMeta.reqId ?? "—"} • 运行：{" "}
            {recMeta.runId ?? "—"}
          </div>
        </div>

        <div className="button-row">
          <button className="btn btn-primary" onClick={handleApplyRecommended}>
            将推荐值应用到输入表
          </button>
          <button className="btn" onClick={handleDownloadJSON}>
            下载推荐结果 JSON
          </button>
          <button className="btn btn-danger" onClick={handleClearRecommendation}>
            清除推荐结果
          </button>
        </div>
      </>
    );
  };

  // ---------- Main render ----------

  return (
    <div className="app-root">
      {/* 🔐 Trial modal overlay */}
      {showTrialModal && (
        <div className="trial-modal-backdrop">
          <div className="trial-modal">
            {trialStage === "choices" && (
              <>
                <h2>试运行模式</h2>
                <p>请选择使用方式：</p>
                <div className="trial-modal-buttons">
                  <button onClick={startTrialFlow} disabled={isStartingTrial}>
                    {isStartingTrial ? "启动中..." : "开始试运行"}
                  </button>
                  <button
                    onClick={() => {
                      // Just checking UI, no active trial; buttons disabled
                      clearTrialTimeout();
                      setTrialStarted(false);
                      setTrialCode(null);
                      setShowTrialModal(false);
                    }}
                  >
                    仅查看界面
                  </button>
                </div>
              </>
            )}

            {trialStage === "code" && trialCode && (
              <>
                <h2>试运行已开始</h2>
                <p>请把以下试运行编号写在纸质记录表上：</p>
                <div className="trial-code-display">{trialCode}</div>
                <p className="trial-hint">记录厚度时请使用同一个编号。</p>
                <button
                  onClick={() => {
                    setShowTrialModal(false);
                  }}
                >
                  好的，已记录
                </button>
              </>
            )}
          </div>
        </div>
      )}

      <div className="top-accent" />
      <header className="topbar">
        <div className="top-left">
          <div className="top-title">OptiParam AI – 试点</div>
          <div className="top-api-compact">
            <label className="api-label">
              API：
              <input
                className="api-input api-input-small"
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
              />
            </label>
          </div>
        </div>

        <div className="top-center">
          {trialStarted && trialCode ? (
            <div className="trial-indicator">
              试运行编号：<span className="trial-indicator-code">{trialCode}</span>
            </div>
          ) : (
            <button
              className="topbar-trial-main-btn"
              onClick={() => {
                clearTrialTimeout();
                setTrialStage("choices");
                setShowTrialModal(true);
              }}
            >
              开始喷涂试运行
            </button>
          )}
        </div>

        <div className="top-right">
          <div>{statusHtml}</div>
        </div>
      </header>

      <div className="tab-row">
        <button
          className={`tab-btn ${view === "Prediction" ? "active" : ""}`}
          onClick={() => setView("Prediction")}
        >
          预测
        </button>
        <button
          className={`tab-btn ${view === "Recommendation" ? "active" : ""}`}
          onClick={() => setView("Recommendation")}
        >
          推荐
        </button>
      </div>

      <main className="main-container">
        <section className="card">
          <div className="card-title">选择机种</div>
          {types.length === 0 ? (
            <div className="caption">没有返回产品类型，请确认数据库已初始化或训练已写入 product_type。</div>
          ) : (
            <select className="select" value={selectedType} onChange={(e) => setSelectedType(e.target.value)}>
              {types.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          )}
        </section>

        <details className="expander">
          <summary>厚度规格 (A–G)</summary>
          <div className="expander-body">{renderSpecsTable()}</div>
        </details>

        {error && <div className="alert alert-error">{error}</div>}

        {view === "Prediction" && (
          <>
            <section className="card">
              <div className="card-title">环境 / 工艺条件</div>
              <div className="grid-3">
                <label className="field">
                  湿度 (%)
                  <input
                    type="number"
                    className="field-input"
                    value={humidity}
                    onChange={(e) => setHumidity(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  温度 (°C)
                  <input
                    type="number"
                    className="field-input"
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  阀门过滤器状态 (Valve_Filter_Status)
                  <input
                    type="number"
                    className="field-input"
                    value={valveStatus}
                    onChange={(e) => setValveStatus(Number(e.target.value))}
                  />
                </label>
              </div>
            </section>

            <section className="card">
              <div className="card-title">喷枪参数</div>
              {renderGunsTable(gunsRows, true)}
            </section>

            <section className="card">
              <div className="card-title">往复机参数</div>
              {renderAxesTable(axesRows, true)}
            </section>

            <div className="button-row">
              <button className="btn" onClick={handleResetMedians}>
                重置为中位值
              </button>
              <button className="btn btn-primary" onClick={handlePredict} disabled={!trialStarted || loadingPredict}>
                {loadingPredict ? "预测中..." : "预测"}
              </button>
            </div>

            {renderPredictionResults()}
          </>
        )}

        {view === "Recommendation" && (
          <>
            <section className="card">
              <div className="card-title">目标厚度 (μm)</div>
              <div className="grid-4">
                {Object.keys(specs)
                  .sort()
                  .map((p) => {
                    const s = specs[p];
                    return (
                      <label className="field" key={p}>
                        {`点位 ${p} 目标(μm) [${s.lo}–${s.hi}]`}
                        <input
                          type="number"
                          className="field-input"
                          value={targets[p] ?? defaultTargets[p] ?? s.target ?? ""}
                          min={s.lo}
                          max={s.hi}
                          onChange={(e) =>
                            setTargets((prev) => ({
                              ...prev,
                              [p]: Number(e.target.value),
                            }))
                          }
                        />
                      </label>
                    );
                  })}
              </div>
            </section>

            <section className="card">
              <div className="card-title">固定条件（优化器不会调整这些）</div>
              <div className="grid-3">
                <label className="field">
                  固定湿度 (%)
                  <input
                    type="number"
                    className="field-input"
                    value={ctxHumidity}
                    onChange={(e) => setCtxHumidity(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  固定温度 (°C)
                  <input
                    type="number"
                    className="field-input"
                    value={ctxTemp}
                    onChange={(e) => setCtxTemp(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  固定阀门过滤器状态 (Valve_Filter_Status)
                  <input
                    type="number"
                    className="field-input"
                    value={ctxValve}
                    onChange={(e) => setCtxValve(Number(e.target.value))}
                  />
                </label>
              </div>
            </section>

            <section className="card">
              <div className="card-title">优化设置</div>
              <div className="grid-3">
                <label className="field">
                  优化次数
                  <input
                    type="number"
                    className="field-input"
                    min={50}
                    max={600}
                    step={25}
                    value={trials}
                    onChange={(e) => setTrials(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  超时 (秒)
                  <input
                    type="number"
                    className="field-input"
                    min={10}
                    max={50}
                    step={5}
                    value={timeoutSec}
                    onChange={(e) => setTimeoutSec(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  相对当前最大步长 (%)
                  <input
                    type="number"
                    className="field-input"
                    min={0}
                    max={20}
                    step={0.5}
                    value={maxStepPct}
                    onChange={(e) => setMaxStepPct(Number(e.target.value))}
                  />
                </label>
              </div>
            </section>

            <div className="button-row">
              <button className="btn btn-primary" onClick={handleRecommend} disabled={!trialStarted || loadingRec}>
                {loadingRec ? "优化中..." : "推荐"}
              </button>
            </div>

            {renderRecommendationSection()}
          </>
        )}

        <hr />
        <div className="caption">使用上方标签切换页面。表格默认使用学习到的中位值；点击“重置为中位值”可恢复。</div>
      </main>
    </div>
  );
};

export default App;
