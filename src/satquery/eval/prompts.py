"""Task prompts, shared by the eval harness and the serving path.

One file so a benchmark number is always produced by the same prompt the deployed
system uses. Changing a prompt changes scores, so treat edits as versioned: bump
``PROMPT_VERSION`` and re-run the affected baselines.
"""

from __future__ import annotations

from satquery.schema import Sample, Task

PROMPT_VERSION = "1.1.0"

#: Generation budget per task. Grounding needs room for a box, captioning for a
#: sentence or two, closed-vocabulary VQA almost nothing.
MAX_NEW_TOKENS: dict[Task, int] = {
    Task.VQA: 32,
    Task.CAPTION: 128,
    Task.GROUNDING: 64,
    Task.CHANGE_VQA: 32,
    Task.CHANGE_CAPTION: 128,
    Task.CROSSMODAL_VQA: 32,
}

_VQA = (
    "This is a remote sensing image. Answer the question using a single word or a "
    "short phrase, with no explanation.\n"
    "Question: {question}\n"
    "Answer:"
)

_CAPTION = (
    "This is a remote sensing image. Describe the land cover, the major objects "
    "present, and their spatial arrangement in one or two sentences. Do not "
    "speculate about anything that is not visible."
)

_GROUNDING = (
    "This is a remote sensing image. Locate the region described below.\n"
    "Respond with only a bounding box in the format [x1, y1, x2, y2], using "
    "integer coordinates normalised to a 0-1000 grid, and nothing else.\n"
    "Description: {question}"
)

_CHANGE_VQA = (
    "These are two remote sensing images of the same geographic area. "
    "Image 1 was acquired earlier and image 2 was acquired later. "
    "Compare them and answer the question using a single word or a short phrase, "
    "with no explanation.\n"
    "Question: {question}\n"
    "Answer:"
)

_CHANGE_CAPTION = (
    "These are two remote sensing images of the same geographic area. "
    "Image 1 was acquired earlier and image 2 was acquired later. "
    "Describe what changed between the two dates and where the change occurred, "
    "in one or two sentences. If nothing changed, say so."
)

# Unlike the other tasks this one is not terse, and deliberately so. The
# single-word instruction elsewhere exists to score under exact match, but no
# public benchmark covers cross-modal analysis -- RSVQA and VRSBench are
# single-image, CDVQA is bi-temporal. This task is judged on the hidden ISRO set
# and demonstrated interactively, where a one-word reply is a non-answer. Asked
# tersely, a model returns the noun phrase from the request and reports nothing.
_CROSSMODAL_VQA = (
    "These are two co-registered remote sensing images of the same geographic "
    "area. Image 1 is optical or multispectral. Image 2 is synthetic aperture "
    "radar (SAR), where water appears very dark and built-up areas appear bright.\n"
    "Answer using both images. State where the built-up and water-covered "
    "regions are, cite the measured fractions given above, and say which sensor "
    "supports each finding. Two or three sentences. Do not repeat the request "
    "back, and do not claim either image is cloudy or obscured unless you can "
    "see that it is.\n"
    "Request: {question}\n"
    "Answer:"
)

_TEMPLATES: dict[Task, str] = {
    Task.VQA: _VQA,
    Task.CAPTION: _CAPTION,
    Task.GROUNDING: _GROUNDING,
    Task.CHANGE_VQA: _CHANGE_VQA,
    Task.CHANGE_CAPTION: _CHANGE_CAPTION,
    Task.CROSSMODAL_VQA: _CROSSMODAL_VQA,
}


def build_prompt(sample: Sample) -> str:
    """Render the prompt for one sample.

    Tasks that take a question require one; captioning tasks ignore it.
    """
    template = _TEMPLATES[sample.task]
    if "{question}" not in template:
        return template
    if not sample.question:
        raise ValueError(f"task {sample.task} requires a question: {sample.sample_id}")
    return template.format(question=sample.question.strip())


def max_new_tokens(task: Task) -> int:
    return MAX_NEW_TOKENS[task]
