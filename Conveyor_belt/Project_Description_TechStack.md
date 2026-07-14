# Smart Bag Counting & Classification System — PoC

## Project Description

### Problem Statement
Manual counting of bags (Semolina, flour, salt, and other packaged products) during truck loading at warehouse bays is error-prone, slow, and disconnected from inventory records. This leads to inventory shrinkage, disputed counts, and no auditable record tying bag counts to specific trucks.

### Objective
Automate bag counting and product-type classification during truck loading using computer vision on existing CCTV camera feeds, eliminating manual counting errors and producing an auditable, timestamped record per truck.

### Core Workflow
```
CCTV Video Stream → AI Object Detection → Object Tracking → Virtual Tripwire Line-Crossing Logic → Live Counter
                                                                        ↓
Operator enters Truck # → clicks START → Database session created → counter increments in real-time
                                                                        ↓
Operator clicks STOP → count finalized → video clip saved → record locked
```

### Scope: PoC vs. Full Production System
This phase is a **proof-of-concept** validating the AI detection/counting/classification core against real sample footage. It is **not** the full production system described in the requirements/questionnaire.

**Proven in this PoC:**
- Bag detection on a moving conveyor belt from top-down/angled CCTV footage
- Multi-frame object tracking (persistent ID per bag, no double-counting)
- Virtual tripwire line-crossing counting logic
- Size-based product-type classification (placeholder rule, pending real-world calibration)
- Annotated output video demonstrating detection + count + classification

**Explicitly out of scope for the PoC (production roadmap items):**
- Live RTSP/ONVIF camera ingestion (PoC uses a recorded video file; swapping to RTSP is a scoped follow-on task)
- START/STOP session UI, truck number entry form
- Database persistence, video clip storage, 30/120-day retention policy
- ERP/WMS/SAP integration
- AD/LDAP authentication, role-based access, audit logs
- Multi-camera (10 concurrent), multi-location, High Availability infrastructure
- Formal accuracy validation against the 98.5–100% NFR target (requires a larger, labeled dataset and real camera calibration, not a single 10–15s clip)

### Key Product Requirements (from source requirements doc)
| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Operator must enter truck ID before session can start | High |
| FR-02 | START initializes stream capture, creates timestamped DB record, resets counter | High |
| FR-03 | CV model counts bags via line-crossing detection | High |
| FR-04 | STOP finalizes count, saves to DB, locks record | High |
| FR-05 | Video evidence clip saved per session, named `[TruckNumber]_[Date]_[Time].mp4` | Medium |

**Non-Functional Requirements (target, full system):**
- Counting accuracy ≥ 98.5% (client discovery answer requested 100%)
- Processing latency ≤ 200ms per frame
- RTSP/ONVIF camera protocol support
- Rolling 120-day video retention (per discovery answers; requirements doc states 30-day)

### Business Context (from discovery questionnaire)
- **Products counted**: Multiple — Semolina, flour, salt, others
- **Bag sizes**: Variable — 10kg, 20kg, 50kg (not fixed per product)
- **Process stage**: Loading only (not unloading)
- **Scale**: 300–1,200 bags/truck, up to ~30,000 bags/day, across multiple locations
- **Cameras**: 6+ installed, RTSP/ONVIF support to be confirmed, indoor, day and night operation
- **AI model training**: Required (per discovery answer) — pretrained/zero-shot detection alone will not meet accuracy targets for multi-product classification
- **Manual override**: Not permitted — system output is treated as final
- **Reporting**: Daily and truck-wise, Excel and PDF export
- **Infrastructure**: On-premises, no GPU server currently available, WiFi connectivity, high availability required

### PoC Validation Findings (from actual sample footage)
- Camera angle in sample clip is oblique top-down (not pure 90° nadir), introducing perspective distortion
- Bags travel with **no gap** between them (touching/adjacent), which is the primary technical risk for per-instance detection accuracy
- Classical CV (Otsu thresholding + ROI masking + connected components) successfully isolates individual bags in most frames but is sensitive to blob fragmentation/fusion, particularly near the camera-near edge of frame
- Multi-frame tracking with tripwire crossing logic is required (not just per-frame counting) to be robust to this single-frame noise — this validates the architecture specified in FR-03 and the vendor's centroid-tracking recommendation

---

## Tech Stack

### PoC (Current Phase)
| Layer | Technology | Notes |
|---|---|---|
| Video I/O | OpenCV (`cv2.VideoCapture`, `cv2.VideoWriter`) | Reads local sample video file |
| Detection | Classical CV: Otsu adaptive thresholding, ROI polygon masking, morphological ops, connected-component analysis | No network access required; robust for controlled, static-background scenes without needing pretrained weights |
| Detection (alternative, requires internet) | YOLOv8/v11 (Ultralytics), zero-shot or fine-tuned | Not testable in current sandboxed environment (no internet egress); viable on a dev machine with connectivity |
| Tracking | Custom centroid tracker (nearest-neighbor association across frames) | Assigns persistent IDs, tolerates a few missed-detection frames |
| Counting logic | Virtual tripwire — line-crossing detection on tracked centroids | Matches FR-03 acceptance criteria |
| Classification | Rule-based size-bucket lookup on bounding-box pixel area | Placeholder bands; requires real-world camera calibration (homography) to be production-accurate |
| Runtime | Python 3, local script execution | No server/API layer in PoC |

### Full Production System (Target Architecture)
| Layer | Technology | Notes |
|---|---|---|
| Frontend | React (web) or Electron (factory-floor desktop app) | Per requirements doc recommendation |
| Backend | Python — FastAPI or Flask | Serves video pipeline, session control, REST API |
| Video ingestion | RTSP / ONVIF via OpenCV or GStreamer | Live multi-camera stream handling |
| CV / Detection | YOLO (fine-tuned on client's actual bag imagery, per-product classes) | Zero-shot pretrained models will not reliably distinguish product types or meet 98.5–100% accuracy target |
| Tracking | Centroid tracker or ByteTrack | Persistent ID across frames for accurate line-crossing counts |
| Database | PostgreSQL (multi-location, ERP-synced) — SQLite viable only for single standalone bay | Discovery confirms multi-location deployment, so PostgreSQL is the realistic choice |
| Storage | Local + Cloud hybrid (per discovery answer) | 120-day rolling retention, compressed `.mp4` clips |
| Integration | ERP/WMS/SAP connector | Required per discovery answers |
| Auth & Security | AD/LDAP integration, role-based access control, audit logging | Required per discovery answers |
| Infrastructure | On-premises deployment, High Availability design, WiFi-based camera connectivity | No GPU server currently available — CPU-based inference constraints or GPU procurement needed |
| Reporting | Excel and PDF export, daily + truck-wise reports | Required per discovery answers |

### Known Gaps / Open Questions Before Production Build
- No GPU server currently available — CPU inference at the required 200ms/frame latency across 10 concurrent camera streams needs feasibility validation or hardware procurement
- Camera make/model, RTSP/ONVIF support, and resolution/FPS specs are unconfirmed ("Will let you know" in discovery answers) — required before finalizing the ingestion layer
- 100% accuracy target with no manual override is a very high bar for computer vision in a variable multi-product, day/night environment; this should be discussed with stakeholders as a risk item, not assumed achievable out of the gate
