"""
simulation_digitaltwins_v4.py
=============================================================
Focus-group simulation using LLM "digital twin" agents.

Workflow (three rounds per experimental run):
  Round 1  – Each agent states an initial standpoint and reasoning.
  Round 2  – Each agent produces a counterargument against a randomly
             selected opposing viewpoint summary.
  Round 3  – Each agent responds to a counterargument that targeted
             its own standpoint.

After all rounds, cosine similarity is computed across runs to measure
how consistent the synthesized viewpoints are.

Usage:
  Edit the USER PARAMETERS block inside main(), then run:
      python simulation_digitaltwins_v3.py

Notes:
  - Use Round1_frequencies.py first to calibrate standpoint frequencies
    against a national baseline (e.g. PEW Research).
  - Summarisation (Rounds 2 & 3) uses a RAG-like approach, so only
    Round 1 frequencies need to be matched to the baseline.
  - Set force=False (default) to reuse cached LLM responses across runs,
    which avoids redundant API calls and reduces cost.
"""

import os
import sys
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('../')
import GoogleGemini as gemini
import GroqLlm as llm
import digitaltwins_promptblocks as dtb

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

rng = np.random.default_rng()

# Load twin personalities once at module level so every function can access them.
_personalities_df = pd.read_csv('twin_personalities_file.csv')
twinpersonalities: dict = _personalities_df.to_dict(orient='list')


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All user-tunable parameters for a simulation run.

    Attributes
    ----------
    models : list[str]
        LLM model identifiers to sample from. Each agent is randomly assigned
        one of these models.
    cov_keys : list[str]
        Column names to use as covariates. The first entry *must* be 'model';
        the remaining entries must be columns present in twin_personalities_file.csv.
    topic : str
        Short description of the discussion topic (no trailing punctuation).
    standpoint_options : list[str]
        Possible positions an agent can take (no trailing punctuation on items).
    background : str
        Factual context provided to each agent before it chooses a standpoint.
    sample_size : int
        Number of LLM agents to simulate per run (must not exceed the number
        of rows in twin_personalities_file.csv).
    strat : bool
        Set to True if using stratified sampling; defaults to False. 
        Important: You can only stratify on the Toubia demographic fields (string values), not the scores. 
        See the DigitalTwinexample.txt file for options. The demographic fields are at the top.
    strat_key: int
        The cov_key for the stratifying variable; only used if strat==True.
        Defaults to 'politics' -- i.e., liberal v conservative
    strat_a: list[str]
        Set the list elements to the A stratification cell -- must be string elements.
    strat_b: list[str]
        Set the list elements to the B stratification cell -- must be string elements.
    num_runs : int
        Number of independent experiment runs (typically 50-100 for traditional
        focus groups, 10-20 for Prytaneum-style).
    save_to_file : bool
        If True, each run's raw responses and summaries are written to disk.
    save_similarities_to_file : bool
        If True, cosine-similarity matrices are written to disk.
    force : bool
        If True, bypass the LLM response cache and always make a fresh API call.
    """
    models: list = field(default_factory=list)
    cov_keys: list = field(default_factory=list)
    topic: str = ''
    standpoint_options: list = field(default_factory=list)
    background: str = ''
    sample_size: int = 8
    strat: bool = False
    strat_key: int = 'politics'
    strat_a: list = field(default_factory=list)
    strat_b: list = field(default_factory=list)
    num_runs: int = 1
    save_to_file: bool = True
    save_similarities_to_file: bool = True
    force: bool = False


# ---------------------------------------------------------------------------
# Private LLM helper functions
# ---------------------------------------------------------------------------

def _call_llm(covdict: dict, prompt: str, force: bool) -> str:
    """Dispatch a prompt to the appropriate LLM backend.

    Routes to Google Gemini if 'gemini' appears in the model name,
    otherwise routes to Groq.

    Parameters
    ----------
    covdict : dict
        Must contain the key 'model' with the model identifier string.
    prompt : str
        The full prompt to send.
    force : bool
        Bypass the response cache when True.

    Returns
    -------
    str
        Raw text response from the LLM.
    """
    if 'gemini' in covdict['model']:
        return gemini.AskGoogleGemini(prompt, model=covdict['model'], force=force)
    return llm.AskGroq(prompt, model=covdict['model'], force=force)


def _call_llm_with_retry(covdict: dict, prompt: str, force: bool, max_retries: int = 5) -> str:
    """Call the LLM and retry up to *max_retries* times on an empty response.

    Parameters
    ----------
    covdict : dict
        Must contain the key 'model'.
    prompt : str
        The full prompt to send.
    force : bool
        Bypass the response cache when True.
    max_retries : int
        Maximum number of attempts before calling sys.exit().

    Returns
    -------
    str
        Non-empty text response from the LLM.
    """
    for attempt in range(1, max_retries + 1):
        response = _call_llm(covdict, prompt, force)
        if len(response) > 0:
            return response
        print(f'Empty response (attempt {attempt}/{max_retries}). Model: {covdict["model"]}')
        print(f'Prompt:\n{prompt}\n-----')

    print('ERROR: All retry attempts exhausted. Exiting.')
    sys.exit(1)


def _strip_qwen_thinking(response: str) -> str:
    """Remove the <think>…</think> block that Qwen models prepend.

    Parameters
    ----------
    response : str
        Raw LLM response, possibly prefixed with a thinking block.

    Returns
    -------
    str
        Response with the thinking block stripped, leading newlines removed.
    """
    if not response.startswith('<think>'):
        return response
    end_tag = '</think>'
    index = response.find(end_tag) + len(end_tag)
    response = response[index:].lstrip('\n')
    return response


def _strip_surrounding_quotes(text: str) -> str:
    """Remove a single layer of surrounding double-quotes if present.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        String with surrounding quotes removed.
    """
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def _extract_json_response(response: str) -> dict:
    """Extract the first JSON object from a raw LLM response string.

    Parameters
    ----------
    response : str
        Raw LLM output that should contain a JSON object.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    json.JSONDecodeError
        If no valid JSON object can be parsed.
    """
    start = response.find('{')
    end = response.find('}') + 1
    return json.loads(response[start:end])


# ---------------------------------------------------------------------------
# Core round functions
# ---------------------------------------------------------------------------

def GetStandpointReasoning(
    covdict: dict,
    topic: str,
    background: str,
    standpoint_options: list[str],
    n: int = 0,
    force: bool = False,
) -> tuple[int, str]:
    """Ask an LLM agent to choose a standpoint and provide its reasoning.

    The agent is given a persona defined by *covdict* and must select one of
    the provided *standpoint_options*, returning a 1-2 sentence justification.

    Parameters
    ----------
    covdict : dict
        Dictionary containing 'model' plus covariate fields (e.g. gender,
        education, politics). Every key must have a corresponding prompt block
        in digitaltwins_promptblocks.py.
    topic : str
        Short description of the discussion topic.
    background : str
        Factual context shown to the agent before it chooses.
    standpoint_options : list[str]
        The labelled standpoints the agent may choose from.
    n : int
        Run index; appended as spaces to the prompt so that each run produces
        a distinct cache entry.
    force : bool
        Bypass the LLM cache when True.

    Returns
    -------
    tuple[int, str]
        (standpoint_number, reasoning) where standpoint_number is 1-indexed.
    """
    # Validate that every covariate has a corresponding prompt block.
    existing_blocks = dtb.existing_prompt_blocks()
    for key in covdict:
        if key not in existing_blocks:
            print(f'ERROR: covariate "{key}" has no prompt block in digitaltwins_promptblocks.py')
            sys.exit(1)

    # Build the prompt.
    prompt = dtb.digitaltwins_blocks(covdict, topic)
    prompt += (
        f'You must select from one of the following possible standpoints on {topic} '
        'and briefly provide your reasoning for doing so in 1-2 sentences. '
    )
    prompt += 'The possible standpoints are:\n'
    for i, standpoint in enumerate(standpoint_options):
        prompt += f'Standpoint {i + 1}: {standpoint.capitalize()}. '
    prompt += ' ' * n   # Cache-busting for individual runs.
    prompt += '\n' + background + '\n'
    if 'qwen' in covdict['model']:
        prompt += '/nothink'  # Reduces inference cost and JSON parse errors.
    prompt += 'Format your response in JSON format such as {"Standpoint 2": "your reasoning"}:\n\n'

    # Attempt to get and parse a JSON response; retry once on failure.
    response = _call_llm(covdict, prompt, force)
    response = _strip_qwen_thinking(response)

    try:
        parsed = _extract_json_response(response)
    except (json.JSONDecodeError, ValueError):
        print('==== JSON parse failed. Raw response:')
        print(response)
        print(f'Model: {covdict["model"]}')
        print('==== Retrying with force=True …')
        response = _call_llm(covdict, prompt, force=True)
        response = _strip_qwen_thinking(response)
        try:
            parsed = _extract_json_response(response)
            print('Resolved.')
            print('-' * 40)
        except (json.JSONDecodeError, ValueError):
            print('==== JSON parse failed again. Raw response:')
            print(response)
            print(f'Model: {covdict["model"]}')
            print('==== Retrying one more time with force=True …')
            response = _call_llm(covdict, prompt, force=True)
            response = _strip_qwen_thinking(response)
            try:
                parsed = _extract_json_response(response)
                print('Resolved.')
                print('-' * 40)
            except (json.JSONDecodeError, ValueError):
                print('PROMPT:')
                print(prompt)
                print('RESPONSE:')
                print(response)
                print(f'MODEL: {covdict["model"]}')
                print('ERROR: Could not extract JSON. Exiting.')
                sys.exit(1)

    # Parse standpoint number and reasoning from the JSON dict.
    raw_key = list(parsed.keys())[0]          # e.g. "Standpoint 2" or "Standpoint 2:"
    standpoint_token = raw_key.split()[1].rstrip(':')
    standpoint_num = int(standpoint_token)
    reasoning = _strip_surrounding_quotes(list(parsed.values())[0])

    return standpoint_num, reasoning


def SummarizeResponses(
    topic: str,
    standpoint: list | str,
    responses: list[str],
    counterargument: bool = False,
    n: int = 0,
    force: bool = False,
) -> list[str]:
    """Summarise a list of participant statements into concise viewpoints.

    This is used after each round to distil many individual statements into
    a compact set of representative summaries that are fed into the next round.

    Parameters
    ----------
    topic : str
        Discussion topic (used in the prompt for context).
    standpoint : list or str
        A single standpoint string (Rounds 1 & 2) or a list of all standpoint
        strings (Round 3, where no single standpoint dominates).
    responses : list[str]
        The raw statements to summarise.
    counterargument : bool
        Set True when *responses* are counterarguments (Round 2), so the
        prompt frames them as arguments *against* the standpoint.
    n : int
        Run index for cache-busting.
    force : bool
        Bypass the LLM cache when True.

    Returns
    -------
    list[str]
        Up to five concise summary strings (one if only one response given).
    """
    if not responses:
        return []

    # Build the framing preamble.
    prompt = (
        f'You are a helpful assistant tasked with summarizing argumentative '
        f'statements on the topic of {topic}. '
        'I have gathered a list of statements that participants in an online '
        'focus group platform have provided as their '
    )

    if isinstance(standpoint, list):
        # Round 3: responses span multiple standpoints.
        standpoints_str = ', '.join(standpoint)
        prompt += (
            f'reasoning for supporting one of the following standpoints: {standpoints_str}. '
            'Some participants may have voiced counterarguments against other viewpoints, '
            'while others may have only stated their own views. '
        )
    else:
        # Rounds 1 & 2: single standpoint.
        if not counterargument:
            prompt += f'reasoning to {standpoint}. '
        else:
            prompt += f'reasoning to not {standpoint}. '

    # Instruct the model on output format.
    if len(responses) == 1:
        prompt += (
            'Summarize the following statements as exactly one concise single summary. '
            'Your summary should be worded as passive tone viewpoints, so that it captures '
        )
    else:
        prompt += (
            'Summarize the following statements as no more than five concise summaries. '
            'Your summaries should be worded as passive tone viewpoints, so that they capture '
        )

    prompt += 'as many viewpoints presented in the given statements as possible. '

    if len(responses) == 1:
        prompt += 'Start the summary on a new line with a dash. '
    else:
        prompt += 'Use as few summaries as are necessary to achieve this. Start each summary on a new line with a dash. '

    prompt += '\n\nThe statements are:\n'
    for i, statement in enumerate(responses):
        prompt += f'Statement {i + 1}: "{statement}"\n'
    prompt += '\nYour summarized viewpoints:\n'
    prompt += ' ' * n  # Cache-busting.

    response = gemini.AskGoogleGemini(prompt, force=force)

    # Parse the dash-separated summaries into a list.
    response = response.replace('\n', '')
    parts = response.split('- ')
    return parts[1:]  # First element is the text before the first dash.


def GetCounterArgument(
    covdict: dict,
    topic: str,
    standpoint: str,
    reasoning: str,
    opposing_viewpoint: str,
    n: int = 0,
    force: bool = False,
) -> str:
    """Ask an LLM agent to produce a counterargument against an opposing view.

    The agent is reminded of its own standpoint and reasoning, then shown a
    summary of the opposing viewpoint and asked to respond critically.

    Parameters
    ----------
    covdict : dict
        Agent's persona and model settings.
    topic : str
        Discussion topic.
    standpoint : str
        The agent's own standpoint from Round 1.
    reasoning : str
        The agent's reasoning from Round 1.
    opposing_viewpoint : str
        A summary of the viewpoint to argue against.
    n : int
        Run index for cache-busting.
    force : bool
        Bypass the LLM cache when True.

    Returns
    -------
    str
        The agent's counterargument (1-2 sentences).
    """
    prompt = dtb.digitaltwins_blocks(covdict, topic)
    prompt += (
        f'You have previously stated that your standpoint is to {standpoint}, '
        f'and your reasoning for this was: "{reasoning}".'
        '\n\n'
        f'As part of the focus group discussion, a different viewpoint has been '
        f'brought up: "{opposing_viewpoint}".'
        '\n'
    )
    prompt += ' ' * n  # Cache-busting.
    prompt += '\n'
    prompt += f'You must now provide a counterargument for this different viewpoint on {topic} in 1-2 sentences:\n'
    if 'qwen' in covdict['model']:
        prompt += '/nothink'

    response = _call_llm_with_retry(covdict, prompt, force, max_retries=3)
    response = _strip_qwen_thinking(response)
    response = _strip_surrounding_quotes(response)
    return response


def GetResponseToCounterArgument(
    covdict: dict,
    topic: str,
    standpoint: str,
    reasoning: str,
    counterargument: str,
    n: int = 0,
    force: bool = False,
) -> str:
    """Ask an LLM agent to respond to a counterargument targeting its view.

    Parameters
    ----------
    covdict : dict
        Agent's persona and model settings.
    topic : str
        Discussion topic.
    standpoint : str
        The agent's own standpoint from Round 1.
    reasoning : str
        The agent's reasoning from Round 1.
    counterargument : str
        The counterargument the agent must respond to.
    n : int
        Run index for cache-busting.
    force : bool
        Bypass the LLM cache when True.

    Returns
    -------
    str
        The agent's response (1-2 sentences).
    """
    prompt = dtb.digitaltwins_blocks(covdict, topic)
    prompt += (
        f'You have previously stated that your standpoint is to {standpoint}, '
        f'and your reasoning for this was: "{reasoning}".'
        '\n\n'
        'As part of the focus group discussion, a counterargument to your standpoint '
        f'has been brought up by a participant who said that: "{counterargument}". '
        '\n\n'
        'You must now respond to this counterargument in 1-2 sentences.\n'
    )
    if 'qwen' in covdict['model']:
        prompt += '/nothink'
    prompt += ' ' * n  # Cache-busting.

    response = _call_llm(covdict, prompt, force)
    response = _strip_qwen_thinking(response)
    return response


# ---------------------------------------------------------------------------
# Experiment sub-functions (modular round runners)
# ---------------------------------------------------------------------------

def _sample_twins(
    models: list[str],
    cov_keys: list[str],
    sample_size: int,
    strat: bool,
    strat_key: int,
    strat_a: list[str],
    strat_b: list[str],
) -> list[dict]:
    """Sample *sample_size* digital twins from the personalities file.

    Each twin is assigned a randomly chosen model and its covariate values
    are drawn without replacement from *twinpersonalities*.

    Parameters
    ----------
    models : list[str]
        Pool of model identifiers to sample from.
    cov_keys : list[str]
        Covariate keys; first entry must be 'model'.
    sample_size : int
        Number of agents to sample.
    strat: bool
        Set to True if using stratified sampling; default to False.
    strat_key: int
        The cov_key that sets the stratifying variable.
    strat_a: list[str]
        List of values for the A stratum. Elements must be string.
    strat_b: list[str]
        List of values for the B stratum. Elements must be string.

    Returns
    -------
    list[dict]
        List of dicts, each containing 'model' plus covariate fields.
    """

    if strat:   # this block is only used if using stratification sampling
        twinpersonalities_a: dict = _personalities_df[_personalities_df[strat_key].isin(strat_a)].to_dict(orient='list')
        twinpersonalities_b: dict = _personalities_df[_personalities_df[strat_key].isin(strat_b)].to_dict(orient='list')
        n_personalities_a = len(twinpersonalities_a['gender'])
        if n_personalities_a < 5:
            print(f'Not enough personas in the {strat_a} strata! Exiting...')
            sys.exit(1)
        n_personalities_b = len(twinpersonalities_b['gender'])
        if n_personalities_b < 5:
            print(f'Not enough personas in the {strat_b} strata! Exiting...')
            sys.exit(1)

        if sample_size/2 % 2 == 0:
            sample_size_a = round(sample_size/2)
            sample_size_b = sample_size - sample_size_a
        else:    # need to randomize which group is larger each time when sample size is odd:
            sample_size_a = int((sample_size + np.random.randint(0,2))/2) # rounds up when random = 1 and down when = 0
            sample_size_b = sample_size - sample_size_a
        if n_personalities_a < sample_size_a:
            print(f'Not enough personas in the {strat_a} strata to sample! Exiting...')
            sys.exit(1)
        if n_personalities_b < sample_size_b:
            print(f'Not enough personas in the {strat_b} strata to sample! Exiting...')
            sys.exit(1)

        sample_indexes_a = np.random.choice(n_personalities_a, sample_size_a, replace=False)
        sample_indexes_b = np.random.choice(n_personalities_b, sample_size_b, replace=False)
        
        agents = []
        for i in sample_indexes_a:
            agent = {'model': models[np.random.randint(0, len(models))]}
            for key in cov_keys[1:]:  # Skip 'model', already set above.
                agent[key] = twinpersonalities_a[key][i]
            agents.append(agent)
        for i in sample_indexes_b:
            agent = {'model': models[np.random.randint(0, len(models))]}
            for key in cov_keys[1:]:  # Skip 'model', already set above.
                agent[key] = twinpersonalities_b[key][i]
            agents.append(agent)
        return agents
        
    else:       # this block is the default, simple random sampling
        n_personalities = len(twinpersonalities['gender'])
        sample_indexes = np.random.choice(n_personalities, sample_size, replace=False)

        agents = []
        for i in sample_indexes:
            agent = {'model': models[np.random.randint(0, len(models))]}
            for key in cov_keys[1:]:  # Skip 'model', already set above.
                agent[key] = twinpersonalities[key][i]
            agents.append(agent)
        return agents


def _run_round1(
    agents: list[dict],
    cov_keys: list[str],
    topic: str,
    background: str,
    standpoint_options: list[str],
    n: int,
    force: bool,
) -> list[dict]:
    """Run Round 1: collect each agent's standpoint and reasoning.

    Mutates *agents* in-place by adding 'standpointNum' and 'reasoning' keys,
    then returns the updated list.

    Parameters
    ----------
    agents : list[dict]
        Sampled agent records (see _sample_twins).
    cov_keys : list[str]
        Covariate keys used to build each agent's covdict.
    topic, background, standpoint_options, n, force :
        Forwarded to GetStandpointReasoning.

    Returns
    -------
    list[dict]
        Agents with 'standpointNum' and 'reasoning' populated.
    """
    print('-' * 80)
    print('Round 1 underway …')
    total = len(agents)

    for i, agent in enumerate(agents):
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total:
            print(f'  Round 1: agent {i + 1}/{total} (run #{n + 1})')

        covdict = {k: agent[k] for k in cov_keys if k in agent}
        standpoint_num, reasoning = GetStandpointReasoning(
            covdict, topic, background, standpoint_options, n, force
        )
        agents[i]['standpointNum'] = standpoint_num
        agents[i]['reasoning'] = reasoning

    return agents


def _summarize_round(
    responses_df: pd.DataFrame,
    standpoint_options: list[str],
    response_col: str,
    group_col: str,
    topic: str,
    is_counterargument: bool,
    n: int,
    force: bool,
) -> dict[int, list[str]]:
    """Summarise responses grouped by standpoint for a single round.

    Parameters
    ----------
    responses_df : pd.DataFrame
        Current responses table.
    standpoint_options : list[str]
        All possible standpoint strings.
    response_col : str
        Column name containing the text to summarise.
    group_col : str
        Column name to group on (standpoint index, 1-based).
    topic : str
        Discussion topic.
    is_counterargument : bool
        Passed through to SummarizeResponses.
    n, force :
        Forwarded to SummarizeResponses.

    Returns
    -------
    dict[int, list[str]]
        Mapping from 0-based standpoint index to list of summary strings.
    """
    summaries: dict[int, list[str]] = {}
    for i, standpoint in enumerate(standpoint_options):
        texts = responses_df.loc[responses_df[group_col] == i + 1, response_col]
        texts = list(texts.values)
        summaries[i] = SummarizeResponses(
            topic, standpoint, texts, is_counterargument, n, force
        )
    return summaries


def _run_round2(
    agents: list[dict],
    cov_keys: list[str],
    topic: str,
    standpoint_options: list[str],
    standpoint_summaries: dict[int, list[str]],
    n: int,
    force: bool,
) -> list[dict]:
    """Run Round 2: each agent produces a counterargument against an opposing view.

    Mutates *agents* in-place by adding 'opposingstandpoint', 'opposingviewpoint',
    and 'counterargument' keys.

    Parameters
    ----------
    agents : list[dict]
        Agents with Round 1 results.
    cov_keys : list[str]
        Covariate keys.
    topic : str
        Discussion topic.
    standpoint_options : list[str]
        All standpoints.
    standpoint_summaries : dict[int, list[str]]
        Summaries from Round 1, keyed by 0-based standpoint index.
    n, force :
        Forwarded to GetCounterArgument.

    Returns
    -------
    list[dict]
        Agents with Round 2 results populated.
    """
    print('-' * 80)
    print('Round 2 underway …')
    total = len(agents)

    for i, agent in enumerate(agents):
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total:
            print(f'  Round 2: agent {i + 1}/{total} (run #{n + 1})')

        # Identify standpoints the agent could argue against (those it didn't choose,
        # that also have at least one summary).
        own_standpoint_idx = agent['standpointNum'] - 1  # 0-based
        counter_candidates = [
            idx for idx, sums in standpoint_summaries.items()
            if idx != own_standpoint_idx and len(sums) > 0
        ]
        # Edge case: unanimous agreement — argue against the agent's own side.
        if not counter_candidates:
            counter_candidates = [own_standpoint_idx]

        counter_idx = np.random.choice(counter_candidates)
        opposing_viewpoint = np.random.choice(standpoint_summaries[counter_idx])

        standpoint = standpoint_options[own_standpoint_idx]
        reasoning = agent['reasoning']
        covdict = {k: agent[k] for k in cov_keys if k in agent}

        counter_arg = GetCounterArgument(
            covdict, topic, standpoint, reasoning, opposing_viewpoint, n, force
        )

        agents[i]['opposingstandpoint'] = counter_idx + 1  # Store as 1-based.
        agents[i]['opposingviewpoint'] = opposing_viewpoint
        agents[i]['counterargument'] = counter_arg

    return agents


def _run_round3(
    agents: list[dict],
    cov_keys: list[str],
    topic: str,
    standpoint_options: list[str],
    opposing_summaries: dict[int, list[str]],
    n: int,
    force: bool,
) -> list[dict]:
    """Run Round 3: each agent responds to a counterargument targeting its view.

    Mutates *agents* in-place by adding 'counterargheard' and 'responsetocounter' keys.

    Parameters
    ----------
    agents : list[dict]
        Agents with Round 1 & 2 results.
    cov_keys : list[str]
        Covariate keys.
    topic : str
        Discussion topic.
    standpoint_options : list[str]
        All standpoints.
    opposing_summaries : dict[int, list[str]]
        Summaries of Round 2 counterarguments, keyed by 0-based standpoint index
        of the standpoint being argued *against*.
    n, force :
        Forwarded to GetResponseToCounterArgument.

    Returns
    -------
    list[dict]
        Agents with Round 3 results populated.
    """
    print('-' * 80)
    print('Round 3 underway …')
    total = len(agents)

    for i, agent in enumerate(agents):
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total:
            print(f'  Round 3: agent {i + 1}/{total} (run #{n + 1})')

        own_standpoint_idx = agent['standpointNum'] - 1  # 0-based
        # opposing_summaries is keyed by the standpoint being argued against.
        counterargs = opposing_summaries.get(own_standpoint_idx, [])

        if counterargs:
            chosen_counter = np.random.choice(counterargs)
            standpoint = standpoint_options[own_standpoint_idx]
            reasoning = agent['reasoning']
            covdict = {k: agent[k] for k in cov_keys if k in agent}
            response = GetResponseToCounterArgument(
                covdict, topic, standpoint, reasoning, chosen_counter, n, force
            )
        else:
            # No counterarguments targeted this standpoint; reuse Round 1 reasoning.
            chosen_counter = agent['reasoning']
            response = agent['reasoning']

        agents[i]['counterargheard'] = chosen_counter
        agents[i]['responsetocounter'] = response

    return agents


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def RunExperiment(
    models: list[str],
    cov_keys: list[str],
    topic: str,
    background: str,
    standpoint_options: list[str],
    directory_name: str,
    sample_size: int,
    strat: bool,
    strat_key: int,
    strat_a: list[str],
    strat_b: list[str],
    n: int = 0,
    save_to_file: bool = False,
    force: bool = False,
) -> dict[int, list[str]]:
    """Run one complete three-round focus-group simulation.

    Samples *sample_size* digital twins, runs them through three discussion
    rounds, and returns the final summarised responses per standpoint.

    Parameters
    ----------
    models : list[str]
        LLM identifiers to randomly assign to agents.
    cov_keys : list[str]
        Covariate keys; first entry must be 'model'.
    topic : str
        Short topic description.
    background : str
        Factual context for agents.
    standpoint_options : list[str]
        Possible standpoints.
    directory_name : str
        Output directory path (used only when save_to_file is True).
    sample_size : int
        Number of agents per run.
    strat: bool
        Set to True of using stratified sampling; defaults to False
    strat_key: int
        The cov_key that sets the stratifying varible
    strat_a: list[str]
        String list of values for the A stratum
    strat_b: list[str]
        String list of values for the B stratum
    n : int
        Run index (0-based); used for cache-busting and output filenames.
    save_to_file : bool
        Write per-run CSVs and JSON summaries when True.
    force : bool
        Bypass LLM cache when True.

    Returns
    -------
    dict[int, list[str]]
        Mapping from 0-based standpoint index to list of final response summaries.
    """
    all_summaries = []  # Accumulates summaries from all three rounds.

    # --- Round 1 ---
    agents = _sample_twins(models, cov_keys, sample_size, strat, strat_key, strat_a, strat_b)
    agents = _run_round1(agents, cov_keys, topic, background, standpoint_options, n, force)

    # Post-Round 1: summarise by standpoint.
    responses_df = pd.DataFrame(agents)
    round1_summaries = _summarize_round(
        responses_df, standpoint_options,
        response_col='reasoning', group_col='standpointNum',
        topic=topic, is_counterargument=False, n=n, force=force,
    )
    all_summaries.append(round1_summaries)

    # --- Round 2 ---
    agents = _run_round2(agents, cov_keys, topic, standpoint_options, round1_summaries, n, force)

    # Post-Round 2: summarise counterarguments by the standpoint being argued against.
    responses_df = pd.DataFrame(agents)
    round2_summaries = _summarize_round(
        responses_df, standpoint_options,
        response_col='counterargument', group_col='opposingstandpoint',
        topic=topic, is_counterargument=True, n=n, force=force,
    )
    all_summaries.append(round2_summaries)

    # --- Round 3 ---
    agents = _run_round3(agents, cov_keys, topic, standpoint_options, round2_summaries, n, force)

    # Post-Round 3: summarise final responses by standpoint (across all standpoints).
    responses_df = pd.DataFrame(agents)
    round3_summaries: dict[int, list[str]] = {}
    for i, standpoint in enumerate(standpoint_options):
        responses = responses_df.loc[responses_df['standpointNum'] == i + 1, 'responsetocounter']
        responses = list(responses.values)
        round3_summaries[i] = SummarizeResponses(
            topic, standpoint_options, responses, False, n, force
        )
    all_summaries.append(round3_summaries)

    # --- Persist results ---
    if save_to_file:
        responses_df.to_csv(f'{directory_name}/responsesDF_n{n}.csv', index=False)
        with open(f'{directory_name}/ResponseSummaries_n{n}.json', 'w', encoding='utf-8') as fh:
            json.dump(all_summaries, fh)

    return round3_summaries


# ---------------------------------------------------------------------------
# Cosine similarity computation
# ---------------------------------------------------------------------------

def compute_cosine_similarities(
    all_standpoint_responses: dict[int, list[str]],
    standpoint_options: list[str],
    directory_name: str,
    save_to_file: bool,
) -> list[float]:
    """Compute cross-run cosine similarity for each standpoint's summaries.

    Encodes each run's merged summary string as a sentence embedding, then
    computes the mean pairwise cosine similarity across runs as a measure of
    how consistent the simulated viewpoints are.

    Parameters
    ----------
    all_standpoint_responses : dict[int, list[str]]
        Mapping from 0-based standpoint index to a list of merged summary
        strings (one per run).
    standpoint_options : list[str]
        Used only for labelling output files.
    directory_name : str
        Output directory for similarity matrices (used when save_to_file is True).
    save_to_file : bool
        Write similarity matrices to CSV when True.

    Returns
    -------
    list[float]
        One mean cosine similarity value per standpoint (standpoints with
        fewer than two responses are skipped).
    """
    print('Calculating cosine similarities …')
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    similarities: list[float] = []

    for standpoint_idx, responses in all_standpoint_responses.items():
        if len(responses) < 2:
            continue  # Need at least two data points for a similarity score.

        embeddings = transformer.encode(responses)
        matrix = cosine_similarity(embeddings)

        # Average the upper triangle (excluding the self-similarity diagonal).
        upper_tri = np.triu_indices_from(matrix, k=1)
        mean_sim = float(np.mean(matrix[upper_tri]))
        similarities.append(mean_sim)

        if save_to_file:
            np.savetxt(
                f'{directory_name}/similarityMatrix_standpoint{standpoint_idx}.csv',
                matrix,
                delimiter=',',
                fmt='%.2f',
            )

    return similarities


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Configure and run the focus-group simulation experiment."""

    # ------------------------------------------------------------------ #
    #                        USER PARAMETERS                              #
    # ------------------------------------------------------------------ #
    cfg = ExperimentConfig(
        models=[
            'gemini-3-flash-preview',
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'qwen/qwen3-32b',
            'openai/gpt-oss-120b',
        ],
        # First entry must be 'model'; remaining entries must be columns
        # in twin_personalities_file.csv.
#        cov_keys=[
#            'model', 'gender', 'education', 'politics',
#            'race', 'pneedforcognition', 'pvc',
#        ],
        cov_keys=[
            'model', 'gender', 'education', 'politics', 'pgreen'
        ],
        # cov_keys=[
        #     'model', 'gender', 'education', 'politics'
        # ],
#        topic='the size of the U.S. House of Representatives. ',
        topic = 'the adoption of Nuclear Power. ',
#        topic = 'gun control in the U.S. ',
#        standpoint_options=[
#            'increase the size of the U.S. House of Representatives by adding more seats',
#            'keep the size of the U.S. House of Representatives the same as it is now',
#        ],
        standpoint_options = [
            'implement rapid expansion of and investment in nuclear power',
            'maintain the current nuclear energy operations without change',
            'phase out existing nuclear plants and halt new construction',
        #    'prioritize researching improved implementations of nuclear power',
        ], 
        # standpoint_options = [
        #     'gun control laws should be more strict',
        #     'gun control laws are about right',
        #     'gun control laws should be less strict',
        # ], 
#        background=(
#            'To help you choose an option, please know that '
#            'expanding the size of the House will reduce the number of constituents '
#            'in each district, enabling representatives to better connect with their '
#            'district residents. However, adding new seats to the House would increase '
#            'the cost of government and potentially make Congress more unwieldy.'
#        ),
        background =(
            'To help you choose an option, please know that according to PEW Research Center, '
            'about half of US residents favor expanding nuclear power, while the other half prefer '
            'to maintain operations without change or to phase it out. Most men '
            'favor expanding it while most women prefer phasing it out or keeping it the same. '
            'Republicans are more likely to favor expanding compared Democrats. '
            'Enviornmentalists have mixed opinions, where some favor expanding it because '
            'nuclear has a low-carbon footprint, while others favor contracting it because they '
            'are concerned about the risks of nuclear waste disposal.'
        ),
        # background =( 
        #     'To help you choose an option, please know that according to the PEW Research Center, '
        #     '82 percent of U.S. conservatives believe that gun control laws should be the same or less strict '
        #     'while 92 percent of U.S. liberals favor making gun control laws more strict. '
        #     'That is, liberals favor making gun control laws more strict and conservatives '
        #     'to keep the laws the same or make them less strict.'
        # ),
        sample_size=10,    # Agents per run; must not exceed rows in personalities file.
        strat=False, # set to True for stratfied sampling. If True, then edit the next two lines.
        strat_key='politics',  # if strat=False, it doesn't matter what is in strat_key, strat_a or strat_b
        strat_a=['Liberal', 'Very liberal'],  # set this to your A stratification cell
        strat_b=['Conservative', 'Very conservative'], # set this to your B stratification cel
        num_runs=1,        # Runs per experiment (50-100 traditional, 10-20 Prytaneum).
        save_to_file=True,
        save_similarities_to_file=True,
        force=False,       # Set True to bypass caching and always make fresh API calls.
    )
    # ------------------------------------------------------------------ #
    #                     END USER PARAMETERS                             #
    # ------------------------------------------------------------------ #

    # Create a uniquely-named output directory for this trial.
    trial_id = str(int(rng.integers(10 ** 10)))
    directory_name = f'Results/trial_id_{trial_id}'
    os.mkdir(directory_name)

    # Initialise LLM clients.
    gemini.InitGoogleGemini()
    llm.InitGroq()

    # Fix the random seed for reproducibility.
    np.random.seed(5591)  # Last four digits of Kevin's office number.

    # Run all experimental rounds, accumulating final summaries across runs.
    all_standpoint_responses: dict[int, list[str]] = {}

    for n in range(cfg.num_runs):
        print('=' * 80)
        print(f'Starting run #{n + 1} of {cfg.num_runs} …')

        standpoint_responses = RunExperiment(
            models=cfg.models,
            cov_keys=cfg.cov_keys,
            topic=cfg.topic,
            background=cfg.background,
            standpoint_options=cfg.standpoint_options,
            directory_name=directory_name,
            sample_size=cfg.sample_size,
            strat=cfg.strat,
            strat_key=cfg.strat_key,
            strat_a=cfg.strat_a,
            strat_b=cfg.strat_b,
            n=n,
            save_to_file=cfg.save_to_file,
            force=cfg.force,
        )

        # Merge each standpoint's run summaries into a single string and store.
        for standpoint_idx, responses in standpoint_responses.items():
            if standpoint_idx not in all_standpoint_responses:
                all_standpoint_responses[standpoint_idx] = []
            all_standpoint_responses[standpoint_idx].append(' '.join(responses))

    print('=' * 80)
    print('Simulation complete. Computing cosine similarities …')

    compute_cosine_similarities(
        all_standpoint_responses,
        cfg.standpoint_options,
        directory_name,
        cfg.save_similarities_to_file,
    )

    # Write parameter settings to disk for reproducibility.
    if cfg.save_to_file:
        if cfg.strat:
            params_text = (
                '**Structured Group Design**\n'
                f'Trial ID number: {trial_id}\n'
                f'Models = {cfg.models}\n'
                f'Covariates = {cfg.cov_keys}\n'
                f'Topic = {cfg.topic}\n'
                f'Standpoint Options = {cfg.standpoint_options}\n'
                f'Background = {cfg.background}\n'
                f'Sample Size = {cfg.sample_size}\n'
                f'Stratification = {cfg.strat}\n'
                f'Stratification Covariate = {cfg.strat_key}\n'
                f'Stratification Cell A = {cfg.strat_a}\n'
                f'Stratification Cell B = {cfg.strat_b}\n'
                f'Number of Experimental Runs = {cfg.num_runs}\n'
            )
        else:
            params_text = (
                '**Structured Group Design**\n'
                f'Trial ID number: {trial_id}\n'
                f'Models = {cfg.models}\n'
                f'Covariates = {cfg.cov_keys}\n'
                f'Topic = {cfg.topic}\n'
                f'Standpoint Options = {cfg.standpoint_options}\n'
                f'Background = {cfg.background}\n'
                f'Sample Size = {cfg.sample_size}\n'
                f'Stratification = {cfg.strat}\n'
                f'Number of Experimental Runs = {cfg.num_runs}\n'
            )
        with open(f'{directory_name}/parameter_settings.txt', 'w') as fh:
            fh.write(params_text)

    print('Done.')


if __name__ == '__main__':
    main()
