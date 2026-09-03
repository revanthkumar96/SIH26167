# Problem Statement: SIH26167

| Field | Detail |
| --- | --- |
| **Problem Statement ID** | SIH26167 |
| **Title** | SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries |
| **Organization** | Indian Space Research Organisation (ISRO) |
| **Department** | Department of Space / ISRO |
| **Category** | Software |
| **Theme** | Space Technology |
| **Hackathon** | Smart India Hackathon 2026 |

---

## Background

Remote-sensing imagery is widely used for agricultural monitoring, disaster management, urban planning, forest monitoring, water-resource assessment, infrastructure mapping, and environmental analysis. However, most existing remote-sensing AI solutions are developed as isolated applications for a single predefined task, such as land-cover classification, object detection, visual question answering, or change detection. These systems often require users to understand satellite-data characteristics, GIS workflows, model selection, and task-specific parameters. Consequently, non-expert users may find it difficult to obtain meaningful information from satellite imagery through simple natural-language queries.

Many operational remote-sensing questions cannot always be answered reliably using a single optical image. Relevant information may be distributed across paired or multiple observations acquired at different times or by different sensors. Optical and multispectral imagery provides spectral and contextual information, whereas synthetic aperture radar (SAR) provides complementary structural information and supports day-and-night acquisition through cloud cover. Multitemporal image pairs are required to identify and interpret changes over time, while co-registered optical–SAR pairs can provide more complete and reliable information than either modality alone.

A general-purpose large language model (LLM) or vision-language model (VLM) cannot be expected to perform these specialised tasks reliably without adaptation to remote-sensing imagery, sensor characteristics, and domain-specific terminology. The proposed solution must therefore include remote-sensing fine-tuning or domain adaptation and may employ multiple specialised models for different tasks.

**BigEarthNet.txt** will serve as the primary dataset for adapting image–text representations to multisensor remote-sensing data. **VRSBench** and **RSVQA** will be used to evaluate single-image captioning, grounding, and visual question answering, while **CDVQA** will be used to evaluate multitemporal change-based visual question answering.

The novelty of SatQuery AI lies in its **agentic, query-driven framework**. Instead of applying a single generic VLM, the system selects and executes suitable remote-sensing specialist models, validates inputs, combines their outputs, and returns an evidence-grounded response.

---

## Objective

Develop **SatQuery AI**, a software-based agentic vision-language assistant for analysing single and paired remote-sensing images through natural-language queries. Single-image understanding is a mandatory baseline. The principal focus is joint reasoning over paired **cross-modal** and **multitemporal** imagery.

---

## Defined Input Scope

| Input type | Description |
| --- | --- |
| **Single image** | One optical/multispectral or SAR image for captioning, visual question answering, and text-guided region grounding. |
| **Cross-modal pair** | Co-registered optical/multispectral and SAR images of the same geographic area for joint information extraction and cross-modal analysis. |
| **Bi-temporal pair** | Two spatially corresponding images of the same geographic area acquired at different times for change detection, change description, and change-based visual question answering. |
| **Supported formats** | GeoTIFF or TIFF for geospatial imagery. PNG and JPEG inputs may be accepted **only** for the prescribed public benchmark datasets. |

---

## Mandatory Functional Scope

1. **Remote-sensing adaptation**  
   At least one visual or vision-language component must be fine-tuned or otherwise adapted using BigEarthNet.txt or any other open-source training data.

2. **Single-image baseline**  
   Visual question answering is mandatory. Each solution must additionally implement **either** captioning / scene description **or** text-guided region grounding.

3. **Multi-image change analysis**  
   Change description or change-based visual question answering from a bi-temporal image pair is mandatory. A spatial change map may also be generated where reference masks are available.

4. **Cross-modal pair analysis**  
   The system must extract complementary information from a co-registered optical/multispectral and SAR image pair.

5. **Agentic orchestration**  
   The system must automatically select, sequence, and execute the appropriate specialist models or tools according to the query and input configuration.

A generic LLM or VLM **without** remote-sensing adaptation will not satisfy the requirements.

---

## Representative Queries

- Describe the land-cover and major objects visible in this image.
- Highlight the water body referred to in the query.
- What changed between these two dates, and where did the change occur?
- Use the optical and SAR images together to identify built-up and water-covered regions.
- Has the built-up area increased, decreased, or remained unchanged?

---

## Agentic Model and Tool Orchestration

The system may use multiple specialised components, such as a remote-sensing VQA or captioning model, a grounding model, a change-understanding or change-VQA model, and an optical–SAR fusion or information-extraction model.

The controller must:

1. Interpret the query and classify the requested task.
2. Check the number, modality, format, metadata, and compatibility of the input images.
3. Select one or more models or tools from a predefined registry.
4. Configure only permitted task parameters and execute the selected workflow.
5. Combine textual and spatial outputs, estimate confidence, and return visual evidence.
6. Provide an auditable execution summary containing the selected task, model/tool names, and key parameters.

The controller may perform internal task planning. **Only the observable execution trace** — selected task, models or tools, permitted parameters, and outputs — will be evaluated. Internal reasoning text is neither required nor evaluated.

---

## Expected Solution

An interactive GUI or web application with an agentic remote-sensing AI backend. It should accept supported image inputs and natural-language queries, select the appropriate specialist workflow, and return evidence-grounded textual and visual results.

The solution should include:

- Input upload and compatibility checking.
- A remote-sensing-adapted vision-language component.
- Specialist tools for VQA, captioning or grounding, change understanding, and optical–SAR analysis.
- An agentic controller for task routing, tool execution, and output integration.
- Visual evidence, confidence information, execution summaries, and downloadable reports.

Each solution must demonstrate:

- Single-image VQA
- One additional single-image task (captioning **or** grounding)
- Multitemporal change understanding
- Optical–SAR paired-image analysis
- Agentic model/tool orchestration

---

## Deliverables

- Interactive GUI or web application with an agentic remote-sensing AI backend
- Source code and models
- Tests and demonstration artefacts

---

## Implementation Scope

The system shall support:

- Single optical/multispectral or SAR images
- Co-registered optical–SAR pairs
- Bi-temporal pairs

in GeoTIFF/TIFF or approved benchmark formats.

It must perform single-image VQA, one additional single-image task, change analysis, optical–SAR joint analysis, and agentic model/tool selection through an interactive GUI or web application.

---

## Evaluation / Judging Criteria

Final evaluation will use prescribed public benchmark test subsets and an **ISRO/SAC** evaluation dataset. Scores will be **normalised before combining** different metrics.

Public benchmarks will be evaluated using the **prescribed test splits**. The ISRO/SAC evaluation set will contain pre-georeferenced and co-registered **Cartosat-2S** optical and **RISAT** SAR image pairs, with task-specific reference answers, labels, bounding boxes, or masks, as applicable. **Evaluation annotations will not be disclosed** to participating teams.

| Criterion | What is evaluated | Dataset / split | Primary metric(s) | Weighting note |
| --- | --- | --- | --- | --- |
| **Single-image VQA** (mandatory) | Answers to natural-language questions on one optical/multispectral or SAR image | VRSBench VQA and RSVQA prescribed test splits | Overall accuracy; per-question-type accuracy where defined | Normalised, then combined with other scores |
| **Additional single-image task — captioning** (choose this *or* grounding) | Scene / land-cover description | VRSBench captioning test split | BLEU, METEOR, ROUGE-L, CIDEr (as prescribed) | Applied only if captioning is the chosen extra task |
| **Additional single-image task — grounding** (choose this *or* captioning) | Text-guided region localisation | VRSBench referring / grounding test split | Acc@IoU 0.5; mIoU (and Acc@IoU 0.25 if prescribed) | Applied only if grounding is the chosen extra task |
| **Multitemporal change understanding** (mandatory) | Change description and/or change-based VQA from a bi-temporal pair | CDVQA official test splits | Overall accuracy (OA); average accuracy (AA) across question types | Spatial change-map IoU / F1 may be used where reference masks exist |
| **Optical–SAR joint analysis** (mandatory) | Complementary information from a co-registered optical–SAR pair (e.g. built-up and water) | ISRO/SAC Cartosat-2S + RISAT pairs; public co-registered pairs (e.g. BigEarthNet.txt) as applicable | Task-specific: answer accuracy, label scores, box IoU, or mask IoU | Hidden ISRO/SAC labels; not available to teams |
| **Agentic orchestration** | Automatic task routing, input checks, tool selection, output fusion | Observable execution trace on demo and evaluation queries | Correct task/tool selection; valid parameters; auditable summary; evidence + confidence | Internal chain-of-thought is **not** scored |
| **System completeness** | GUI/web app, uploads, compatibility checks, visual evidence, reports | Demonstration | Functional checklist against mandatory scope | Required for a valid submission |

Scores from heterogeneous metrics (accuracy, caption n-gram scores, IoU) are **normalised** before aggregation so that no single metric dominates by scale.

---

## Datasets

All listed public datasets are available online as open-source resources. Use **official / prescribed test splits** for evaluation.

### Training / fine-tuning (primary)

| Dataset | Role | Link |
| --- | --- | --- |
| **BigEarthNet.txt** | Primary dataset for remote-sensing adaptation using co-registered Sentinel-1 SAR, Sentinel-2 multispectral imagery, and diverse text annotations (captions, VQA, referring expressions) | [arXiv:2603.29630](https://arxiv.org/abs/2603.29630) |

Other open-source remote-sensing training data may be used in addition, provided at least one visual or vision-language component is adapted as required.

### Public evaluation benchmarks

| Dataset | Role | Link |
| --- | --- | --- |
| **VRSBench** | Single-image captioning, text-guided grounding, and visual question answering | [vrsbench.github.io](https://vrsbench.github.io/) · [arXiv:2406.12384](https://arxiv.org/abs/2406.12384) · [GitHub](https://github.com/lx709/VRSBench) |
| **RSVQA** | Single-image visual question answering on remote-sensing imagery | [RSVQA project](https://rsvqa.sylvainlobry.com/) · [GitHub](https://github.com/syvlo/RSVQA) |
| **CDVQA** | Multitemporal change-based visual question answering | [GitHub](https://github.com/YZHJessica/CDVQA) |

### Hidden evaluation set (organisers)

| Dataset | Role |
| --- | --- |
| **ISRO/SAC evaluation set** | Pre-georeferenced, co-registered Cartosat-2S optical and RISAT SAR image pairs with undisclosed task-specific references (answers, labels, boxes, or masks) |

---

## Constraints and disqualifiers

- PNG/JPEG are allowed only for prescribed public benchmark datasets; operational geospatial inputs should be GeoTIFF/TIFF.
- Only permitted task parameters may be configured by the agent.
- A generic LLM/VLM with no remote-sensing fine-tuning or domain adaptation does **not** meet the problem requirements.
- Evaluation of the agent uses the **observable execution trace**, not hidden internal reasoning.
