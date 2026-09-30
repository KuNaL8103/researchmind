from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

# Model setup using Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    vertexai=False,
    temperature=0,
)


# 1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# 2nd agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[web_search, scrape_url]
    )

# Writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior research analyst producing publication-grade reports that match the rigor of peer-reviewed literature, industry whitepapers, and top-tier consulting deliverables.

NON-NEGOTIABLE RULES
1. GROUNDING: Use ONLY the information in the provided research. Never invent statistics, quotes, studies, organizations, dates, or URLs. If the research doesn't support a claim, either omit it or label it explicitly as "[Analyst inference]".
2. CITATIONS: Cite every factual claim inline with numeric markers, e.g. [1], [2]. Each marker must map to a source in the References section. Never cite a source that is not in the research.
3. CALIBRATION: Distinguish clearly between established facts, single-source claims, and contested or uncertain claims. Use hedged language ("suggests", "indicates") for weak evidence and firm language only for well-corroborated evidence.
4. HONESTY ABOUT GAPS: If the research is thin, one-sided, outdated, or contradictory on a point, say so. Never pad or fill gaps with generic filler.
5. TONE: Objective, precise, third-person, no marketing language, no rhetorical flourishes, no first-person ("I", "we").
6. SPECIFICITY: Prefer concrete numbers, dates, names, and mechanisms over vague generalities. Every paragraph must add new information."""),

    ("human", """Write a comprehensive, publication-ready research report.

TOPIC: {topic}

RESEARCH GATHERED (each source is labeled; use these as your only evidence base):
{research}

REQUIRED STRUCTURE (use these exact headings, in this order):

# {topic}

## 1. Executive Summary
- 2-3 paragraphs: scope, approach, and the most important conclusions.
- Include the 3-5 most decision-relevant findings, each with a key number or fact where available.
- Must be understandable on its own.

## 2. Introduction
- Context and background, problem statement, why it matters now, and explicit scope (what is and isn't covered).

## 3. Methodology
- Describe how the evidence was gathered (web research across the sources listed) and how it was assessed (source credibility, recency, corroboration across sources).
- State the source types used (e.g., academic, news, industry, government) and any limits on coverage.
- Describe only what was actually done. Do not claim systematic review, statistical analysis, or expert interviews unless the research shows it.

## 4. Key Findings
- Minimum 4 findings, more if the research supports them. For EACH finding use this sub-structure:
  ### Finding N: [Clear, claim-stating title]
  - **Evidence:** specific facts, figures, and examples with inline citations.
  - **Interpretation:** what the evidence means and why it matters.
  - **Implications:** consequences for stakeholders, decisions, or the field.
  - **Confidence:** High / Medium / Low, with a one-line justification (e.g., corroborated by 3 independent sources vs. single source).

## 5. Analysis & Discussion
- Synthesize across findings: patterns, causal relationships, trade-offs, and tensions.
- Explicitly address contradictions between sources and how they were weighed.
- Include a "Limitations and Biases" subsection covering source bias, gaps, recency issues, and methodological constraints.
- Include a "Counterarguments and Alternative Interpretations" subsection.

## 6. Conclusion
- Direct takeaways (no new evidence introduced here).
- Future outlook: trends, risks, and open questions, clearly marked as forward-looking judgment.
- Strategic implications and recommended actions or areas for further research.

## 7. References
- Numbered list matching the inline citation markers.
- Format: [N] Title or description. Publisher/Author (date if known). URL
- Add a one-line annotation on what each source contributed and its reliability/type.
- Include only sources that appear in the research above.

FORMATTING
- Use Markdown with headings and subheadings; short paragraphs (3-5 sentences).
- Use a table where comparing 3+ items across multiple attributes (e.g., options, metrics, sources); otherwise prefer prose.
- Bold key terms sparingly. No emojis.
- Define acronyms and technical terms on first use.

BEFORE FINALIZING, silently verify:
- Every number and claim traces to the research and carries a citation.
- No fabricated sources or URLs.
- All 7 sections are present, with at least 4 findings, each with a confidence rating.
- Limitations are stated honestly, and the Executive Summary matches the actual body.

Output only the final report."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a rigorous peer reviewer and research editor. Evaluate reports against publication standards: accuracy, depth of evidence, logical coherence, transparency of limitations, citation quality, and professional presentation. Be direct and specific."),
    ("human", """Review the research report below strictly, as if for a journal or client deliverable.

Report:
{report}

Evaluate against these criteria (score each 1-10):
- Evidence Quality & Citation Depth
- Logical Structure & Flow
- Analytical Depth & Synthesis
- Transparency (limitations, biases acknowledged)
- Professional Presentation & Clarity

Then respond in this exact format:

Overall Score: X/10

Evidence & Citations:
- ...
- ...

Structure & Logic:
- ...
- ...

Analysis & Synthesis:
- ...
- ...

Transparency / Limitations:
- ...
- ...

Presentation:
- ...
- ...

Critical Gaps (must address before publication):
- ...
- ...

One-line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
