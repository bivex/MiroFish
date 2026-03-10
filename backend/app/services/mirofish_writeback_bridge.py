"""Optional bridge from completed MiroFish reports to loreSystem write-back API."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from ..config import Config
from ..models.project import ProjectManager
from ..utils.logger import get_logger
from .simulation_manager import SimulationManager
from .simulation_runner import SimulationRunner

logger = get_logger('mirofish.writeback_bridge')

_SECTION_RE = re.compile(r'^##\s+(Actors|Organizations)\s*$')
_ENTRY_RE = re.compile(r'^(Actor|Organization):\s*(.+?)\s*$')
_FIELD_RE = re.compile(r'^-\s+([a-zA-Z0-9_]+):\s*(.*)$')
_META_RE = re.compile(r'^(schema_version|world_id|world_version|scenario_id):\s*(.+?)\s*$')
_RUMOR_RE = re.compile(r'\b(rumou?r|claim|alleg(?:ation|ed)?|speculat(?:ion|e|ive)|whisper|gossip|deny|denial)\b', re.IGNORECASE)
_NEGATIVE_RELATIONSHIP_RE = re.compile(r'\b(blame|distrust|accus|critic|clash|conflict|tension|feud|threat|denounce)\b', re.IGNORECASE)
_POSITIVE_RELATIONSHIP_RE = re.compile(r'\b(ally|support|coordina(?:te|tion)|assist|back|trust|protect|cooperate)\b', re.IGNORECASE)
_CANONICAL_ID_OFFSETS = {
    'Character': 100000,
    'Faction': 200000,
    'Organization': 200000,
    'Location': 300000,
    'Event': 400000,
    'Rumor': 500000,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_iso(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return _utc_now_iso()
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _norm(value: Any) -> str:
    return ' '.join(str(value or '').strip().lower().split())


def _safe_id(prefix: str, value: Any) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', _norm(value)).strip('_') or 'unknown'
    return f'{prefix}:{slug}'


def _stable_hex(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    return f'{prefix}-{hashlib.sha256(encoded).hexdigest()[:24]}'


def _stable_numeric_id(value: Any, *, canonical_type: str | None = None) -> int:
    text = str(value or '').strip()
    offset = _CANONICAL_ID_OFFSETS.get(str(canonical_type or '').strip(), 900000)
    if text:
        match = re.search(r'(\d+)$', text)
        if match:
            return offset + int(match.group(1))
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest() if text else hashlib.sha256(b'unknown').hexdigest()
    return offset + (int(digest[:8], 16) % 90000)


def _extract_projection_metadata(text: str | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in (text or '').splitlines():
        line = raw_line.strip()
        if not line:
            break
        match = _META_RE.match(line)
        if match:
            metadata[match.group(1)] = match.group(2).strip()
    return metadata


def _parse_projection_subjects(text: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actors: list[dict[str, Any]] = []
    organizations: list[dict[str, Any]] = []
    current_section: str | None = None
    current_entry: dict[str, Any] | None = None

    def flush_entry() -> None:
        nonlocal current_entry
        if not current_entry:
            return
        payload = dict(current_entry)
        payload['id'] = payload.get('id') or _safe_id(current_section or 'actor', payload.get('name'))
        if current_section == 'actor':
            actors.append(payload)
        elif current_section == 'organization':
            organizations.append(payload)
        current_entry = None

    for raw_line in list((text or '').splitlines()) + ['## __END__']:
        line = raw_line.strip()
        section_match = _SECTION_RE.match(line)
        if section_match or line == '## __END__':
            flush_entry()
            current_section = {'Actors': 'actor', 'Organizations': 'organization'}.get(section_match.group(1)) if section_match else None
            continue
        if current_section not in {'actor', 'organization'}:
            continue
        entry_match = _ENTRY_RE.match(line)
        if entry_match:
            flush_entry()
            current_entry = {'name': entry_match.group(2).strip()}
            continue
        field_match = _FIELD_RE.match(line)
        if field_match and current_entry is not None:
            current_entry[field_match.group(1)] = field_match.group(2).strip()
    return actors, organizations


def _profiles_to_actor_subjects(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    for profile in profiles:
        name = str(profile.get('name') or profile.get('realname') or profile.get('username') or profile.get('user_name') or '').strip()
        if not name:
            continue
        subject_ref = str(
            profile.get('represented_entity_id')
            or profile.get('id')
            or profile.get('subject_ref')
            or _safe_id('actor', profile.get('username') or profile.get('user_name') or name)
        )
        subjects.append({
            'id': subject_ref,
            'name': name,
            'canonical_id': profile.get('source_entity_uuid') or profile.get('canonical_id'),
            'canonical_type': profile.get('source_entity_type') or profile.get('canonical_type'),
            'speaker_mode': profile.get('speaker_mode') or 'individual',
            'represented_entity_id': profile.get('represented_entity_id'),
            'username': profile.get('username') or profile.get('user_name'),
            'description': profile.get('description') or profile.get('bio') or profile.get('profile'),
        })
    return subjects


def _dedupe_subjects(subjects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        subject_ref = str(subject.get('id') or '').strip()
        if not subject_ref:
            continue
        deduped.setdefault(subject_ref, subject)
    return list(deduped.values())


def _build_subject_lookup(subjects: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for subject in subjects:
        subject_ref = str(subject.get('id') or '').strip()
        if not subject_ref:
            continue
        for key in ('name', 'username', 'user_name', 'realname'):
            value = subject.get(key)
            if _norm(value):
                lookup[_norm(value)] = subject_ref
    return lookup


def _build_evidence_text(action: Any) -> str:
    base = f'[{action.platform}][round {action.round_num}] {action.agent_name} {action.action_type}'
    detail = action.result
    if not detail and getattr(action, 'action_args', None):
        detail = json.dumps(action.action_args, ensure_ascii=False, sort_keys=True)
    return f'{base}: {detail}'.strip(': ')[:2000]


def _action_detail_text(action: Any) -> str:
    detail = str(getattr(action, 'result', '') or '').strip()
    if detail:
        return detail
    args = getattr(action, 'action_args', None)
    if isinstance(args, dict):
        for key in ('post', 'content', 'text', 'body', 'message', 'comment', 'title'):
            value = args.get(key)
            if str(value or '').strip():
                return str(value).strip()
    if args:
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    return ''


def _trim_text(value: Any, *, limit: int = 240) -> str:
    text = ' '.join(str(value or '').strip().split())
    return text[:limit]


def _actor_directory(actors: list[dict[str, Any]]) -> list[dict[str, str]]:
    directory: list[dict[str, str]] = []
    for actor in actors:
        subject_ref = str(actor.get('id') or '').strip()
        name = str(actor.get('name') or '').strip()
        if subject_ref and name:
            directory.append({'ref': subject_ref, 'name': name, 'norm_name': _norm(name)})
    return directory


def _actor_mentions(text: str, actor_directory: list[dict[str, str]], *, exclude_ref: str | None = None) -> list[dict[str, str]]:
    normalized_text = _norm(text)
    if not normalized_text:
        return []
    matches: list[dict[str, str]] = []
    for actor in actor_directory:
        if exclude_ref and actor['ref'] == exclude_ref:
            continue
        if actor['norm_name'] and actor['norm_name'] in normalized_text:
            matches.append(actor)
    return matches


def _relationship_level_from_text(text: str) -> int:
    if _NEGATIVE_RELATIONSHIP_RE.search(text or ''):
        return -36
    if _POSITIVE_RELATIONSHIP_RE.search(text or ''):
        return 34
    return 5


def _append_collection_evidence(
    runtime_evidence: list[dict[str, Any]],
    items: list[dict[str, Any]],
    *,
    world_id: str,
    scenario_id: str,
    run_id: str,
    report: Any,
    collection_name: str,
    evidence_type: str,
    source_type: str,
) -> None:
    for index, item in enumerate(items):
        evidence_payload = {
            'collection': collection_name,
            'index': index,
            'run_id': run_id,
            'timestamp': _utc_iso(item.get('timestamp') or report.completed_at),
            'summary': _trim_text(item.get('summary') or item.get('description') or item.get('name')),
        }
        runtime_evidence.append({
            'evidence_id': _stable_hex('ev', evidence_payload),
            'world_id': world_id,
            'scenario_id': scenario_id,
            'run_id': run_id,
            'evidence_type': evidence_type,
            'source_type': source_type,
            'actor_refs': item.get('actor_refs') or item.get('participant_ids') or [],
            'text': _trim_text(item.get('summary') or item.get('description') or item.get('name')),
            'structured_payload': item,
            'timestamp': _utc_iso(item.get('timestamp') or report.completed_at),
            'confidence': item.get('confidence', 0.6),
            'source_refs': [
                {'collection': collection_name, 'index': index},
                {'type': 'simulation', 'simulation_id': report.simulation_id},
                {'type': 'report', 'report_id': report.report_id},
                {'type': 'graph', 'graph_id': report.graph_id},
            ],
        })


def _index_collection_evidence_ids(runtime_evidence: list[dict[str, Any]]) -> dict[tuple[str, int], list[str]]:
    evidence_map: dict[tuple[str, int], list[str]] = {}
    for item in runtime_evidence:
        evidence_id = str(item.get('evidence_id') or '').strip()
        if not evidence_id:
            continue
        for ref in item.get('source_refs') or []:
            collection = str(ref.get('collection') or '').strip()
            if not collection:
                continue
            index = ref.get('index')
            if not isinstance(index, int):
                continue
            evidence_map.setdefault((collection, index), []).append(evidence_id)
    return evidence_map


def _subject_numeric_map(subjects: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for subject in subjects:
        subject_ref = str(subject.get('id') or '').strip()
        if not subject_ref:
            continue
        mapping[subject_ref] = _stable_numeric_id(
            subject.get('canonical_id') or subject_ref,
            canonical_type=str(subject.get('canonical_type') or ''),
        )
    return mapping


def _candidate_record(
    *,
    world_id: str,
    scenario_id: str,
    run_id: str,
    candidate_type: str,
    target_canonical_type: str,
    name: str,
    summary: str,
    proposed_change: dict[str, Any],
    evidence_ids: list[str],
    source_refs: list[dict[str, Any]],
    confidence: float,
) -> dict[str, Any]:
    payload = {
        'world_id': world_id,
        'scenario_id': scenario_id,
        'run_id': run_id,
        'candidate_type': candidate_type,
        'target_canonical_type': target_canonical_type,
        'name': name,
        'summary': summary,
        'proposed_change': proposed_change,
        'evidence_ids': evidence_ids,
        'source_refs': source_refs,
        'confidence': confidence,
        'status': 'pending_review',
    }
    return {'candidate_id': _stable_hex('cand', payload), **payload}


def _build_candidate_deltas(
    *,
    world_id: str,
    scenario_id: str,
    run_id: str,
    emergent_events: list[dict[str, Any]],
    rumor_candidates: list[dict[str, Any]],
    relationship_changes: list[dict[str, Any]],
    runtime_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collection_evidence = _index_collection_evidence_ids(runtime_evidence)
    action_evidence_ids = [
        str(item.get('evidence_id') or '').strip()
        for item in runtime_evidence
        if str(item.get('source_type') or '').strip() == 'mirofish_action_log' and str(item.get('evidence_id') or '').strip()
    ]

    def support_ids(*keys: tuple[str, int]) -> list[str]:
        ids: list[str] = []
        for key in keys:
            ids.extend(collection_evidence.get(key, []))
        ids.extend(action_evidence_ids[:2])
        deduped: list[str] = []
        seen: set[str] = set()
        for evidence_id in ids:
            if evidence_id and evidence_id not in seen:
                deduped.append(evidence_id)
                seen.add(evidence_id)
        return deduped

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(emergent_events):
        evidence_ids = support_ids(('emergent_event', index))
        candidates.append(
            _candidate_record(
                world_id=world_id,
                scenario_id=scenario_id,
                run_id=run_id,
                candidate_type='scenario_event',
                target_canonical_type='Event',
                name=str(item.get('name') or 'Simulation event').strip() or 'Simulation event',
                summary=str(item.get('summary') or item.get('description') or item.get('name') or 'Simulation event').strip(),
                proposed_change=dict(item),
                evidence_ids=evidence_ids,
                source_refs=[{'collection': 'emergent_event', 'index': index}],
                confidence=max(float(item.get('confidence') or 0.0), 0.92 if len(evidence_ids) >= 2 else 0.78),
            )
        )

    for index, item in enumerate(rumor_candidates):
        evidence_ids = support_ids(('rumor_candidate', index), ('prediction_summary', index))
        candidates.append(
            _candidate_record(
                world_id=world_id,
                scenario_id=scenario_id,
                run_id=run_id,
                candidate_type='rumor_candidate',
                target_canonical_type='Rumor',
                name=str(item.get('name') or 'Simulation rumor').strip() or 'Simulation rumor',
                summary=str(item.get('summary') or item.get('description') or item.get('name') or 'Simulation rumor').strip(),
                proposed_change=dict(item),
                evidence_ids=evidence_ids,
                source_refs=[{'collection': 'rumor_candidate', 'index': index}, {'collection': 'prediction_summary', 'index': index}],
                confidence=max(float(item.get('confidence') or 0.0), 0.91 if len(evidence_ids) >= 2 else 0.72),
            )
        )

    for index, item in enumerate(relationship_changes):
        evidence_ids = support_ids(('relationship_delta', index))
        candidates.append(
            _candidate_record(
                world_id=world_id,
                scenario_id=scenario_id,
                run_id=run_id,
                candidate_type='relationship_change',
                target_canonical_type='CharacterRelationship',
                name=str(item.get('name') or 'Simulation relationship change').strip() or 'Simulation relationship change',
                summary=str(item.get('summary') or item.get('description') or item.get('name') or 'Simulation relationship change').strip(),
                proposed_change=dict(item),
                evidence_ids=evidence_ids,
                source_refs=[{'collection': 'relationship_delta', 'index': index}],
                confidence=float(item.get('confidence') or 0.62),
            )
        )
    return candidates


def _build_auto_promote_requests(bundle: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    tenant_id = Config.MIROFISH_WRITEBACK_AUTO_PROMOTE_TENANT_ID
    world_id = Config.MIROFISH_WRITEBACK_AUTO_PROMOTE_WORLD_ID
    if tenant_id is None or world_id is None:
        raise ValueError('Auto-promotion requires MIROFISH_WRITEBACK_AUTO_PROMOTE_TENANT_ID and MIROFISH_WRITEBACK_AUTO_PROMOTE_WORLD_ID')

    subjects = [*(bundle.get('actors') or []), *(bundle.get('organizations') or [])]
    subject_numeric_map = _subject_numeric_map(subjects)
    grouped: dict[str, list[dict[str, Any]]] = {}

    for candidate in bundle.get('candidate_deltas') or []:
        candidate_id = str(candidate.get('candidate_id') or '').strip()
        proposed = candidate.get('proposed_change') or {}
        candidate_type = str(candidate.get('candidate_type') or '').strip()
        confidence = float(candidate.get('confidence') or 0.0)
        evidence_ids = [str(item).strip() for item in (candidate.get('evidence_ids') or []) if str(item).strip()]
        if not candidate_id or confidence < 0.90 or len(evidence_ids) < 2:
            continue
        if candidate_type == 'scenario_event':
            participant_map = {
                str(ref): subject_numeric_map[str(ref)]
                for ref in (proposed.get('participant_ids') or [])
                if str(ref) in subject_numeric_map
            }
            if not participant_map:
                continue
            grouped.setdefault('safe_event_only', []).append({
                'candidate_id': candidate_id,
                'mapping': {
                    'tenant_id': tenant_id,
                    'world_id': world_id,
                    'participant_map': participant_map,
                },
            })
        elif candidate_type == 'rumor_candidate':
            source_name = str(proposed.get('source_name') or '').strip() or 'MiroFish report synthesis'
            grouped.setdefault('safe_rumor_only', []).append({
                'candidate_id': candidate_id,
                'mapping': {
                    'tenant_id': tenant_id,
                    'world_id': world_id,
                    'source_name': source_name,
                    'credibility_score': max(7, min(10, int(round(confidence * 10)))),
                    'truth_level': str(proposed.get('truth_level') or 'Unverified'),
                    'spread_speed': str(proposed.get('spread_speed') or 'Moderate'),
                },
            })
        elif candidate_type == 'relationship_change':
            actor_refs = [str(ref) for ref in (proposed.get('actor_refs') or []) if str(ref) in subject_numeric_map]
            relationship_level = int(proposed.get('relationship_level') or 0)
            if len(actor_refs) != 2 or abs(relationship_level) < 30:
                continue
            grouped.setdefault('safe_relationship_only', []).append({
                'candidate_id': candidate_id,
                'mapping': {
                    'tenant_id': tenant_id,
                    'world_id': world_id,
                    'character_from_id': subject_numeric_map[actor_refs[0]],
                    'character_to_id': subject_numeric_map[actor_refs[1]],
                    'relationship_level': relationship_level,
                    'is_mutual': False,
                },
            })

    return [(policy, items) for policy, items in grouped.items() if items]


def _derive_rumor_candidates(
    *,
    actions: list[Any],
    report: Any,
    actors: list[dict[str, Any]],
    organizations: list[dict[str, Any]],
    subject_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    rumor_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    fallback_refs = [
        str(item.get('id') or '').strip()
        for item in (organizations + actors)
        if str(item.get('id') or '').strip()
    ]

    for action in actions:
        detail = _action_detail_text(action)
        if not _RUMOR_RE.search(detail):
            continue
        matched_ref = subject_lookup.get(_norm(action.agent_name))
        summary = _trim_text(detail, limit=320)
        signature = _norm(summary)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        rumor_candidates.append({
            'name': _trim_text(f'Rumor signal from {action.agent_name}', limit=120),
            'summary': summary,
            'text': summary,
            'actor_refs': [matched_ref] if matched_ref else [],
            'source_name': str(action.agent_name or '').strip() or None,
            'timestamp': _utc_iso(action.timestamp),
            'confidence': 0.72 if getattr(action, 'success', True) else 0.55,
            'platform': action.platform,
            'action_type': action.action_type,
        })
        if len(rumor_candidates) >= 3:
            break

    summary_text = str(getattr(getattr(report, 'outline', None), 'summary', '') or '').strip()
    if not rumor_candidates and _RUMOR_RE.search(summary_text):
        rumor_candidates.append({
            'name': _trim_text(f'Rumor surfaced in {getattr(getattr(report, "outline", None), "title", "report")}', limit=120),
            'summary': _trim_text(summary_text, limit=320),
            'text': _trim_text(summary_text, limit=320),
            'actor_refs': fallback_refs[:1],
            'source_name': getattr(getattr(report, 'outline', None), 'title', None),
            'timestamp': _utc_iso(report.completed_at),
            'confidence': 0.64,
        })
    return rumor_candidates


def _derive_emergent_events(
    *,
    actions: list[Any],
    report: Any,
    actors: list[dict[str, Any]],
    organizations: list[dict[str, Any]],
    subject_lookup: dict[str, str],
    project: Any,
) -> list[dict[str, Any]]:
    participants: list[str] = []
    seen_participants: set[str] = set()
    for action in actions:
        matched_ref = subject_lookup.get(_norm(action.agent_name))
        if matched_ref and matched_ref not in seen_participants:
            seen_participants.add(matched_ref)
            participants.append(matched_ref)
    for item in organizations:
        subject_ref = str(item.get('id') or '').strip()
        if subject_ref and subject_ref not in seen_participants:
            seen_participants.add(subject_ref)
            participants.append(subject_ref)
            break

    summary = _trim_text(
        getattr(getattr(report, 'outline', None), 'summary', None)
        or getattr(report, 'simulation_requirement', None)
        or getattr(project, 'analysis_summary', None)
        or f'{len(actions)} actions observed during simulation',
        limit=320,
    )
    if not summary:
        return []

    return [{
        'name': _trim_text(getattr(getattr(report, 'outline', None), 'title', None) or f'{project.name} emergent event', limit=120),
        'summary': summary,
        'description': summary,
        'participant_ids': participants,
        'actor_refs': participants,
        'timestamp': _utc_iso(actions[-1].timestamp if actions else report.completed_at),
        'confidence': 0.7 if actions else 0.6,
        'outcome': 'ongoing',
    }]


def _derive_relationship_changes(*, actions: list[Any], actors: list[dict[str, Any]], subject_lookup: dict[str, str]) -> list[dict[str, Any]]:
    actor_directory = _actor_directory(actors)
    relationship_changes: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for action in actions:
        source_ref = subject_lookup.get(_norm(action.agent_name))
        if not source_ref:
            continue
        detail = _action_detail_text(action)
        for target in _actor_mentions(detail, actor_directory, exclude_ref=source_ref):
            pair = tuple(sorted((source_ref, target['ref'])))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            summary = _trim_text(detail, limit=320)
            relationship_level = _relationship_level_from_text(detail)
            relationship_changes.append({
                'name': _trim_text(f'{action.agent_name} reacts to {target["name"]}', limit=120),
                'summary': summary,
                'text': summary,
                'actor_refs': [source_ref, target['ref']],
                'character_from_ref': source_ref,
                'character_to_ref': target['ref'],
                'relationship_level': relationship_level,
                'relationship_type': 'enemy' if relationship_level < 0 else ('friend' if relationship_level > 8 else 'neutral'),
                'is_mutual': False,
                'timestamp': _utc_iso(action.timestamp),
                'confidence': 0.62,
            })
            if len(relationship_changes) >= 2:
                return relationship_changes
    return relationship_changes


def load_context(report: Any, *, graph_backend: str | None = None) -> dict[str, Any]:
    simulation_manager = SimulationManager()
    simulation_state = simulation_manager.get_simulation(report.simulation_id)
    if not simulation_state:
        raise ValueError(f'Simulation not found: {report.simulation_id}')

    project = ProjectManager.get_project(simulation_state.project_id)
    if not project:
        raise ValueError(f'Project not found: {simulation_state.project_id}')

    extracted_text = ProjectManager.get_extracted_text(project.project_id)
    projection_meta = _extract_projection_metadata(extracted_text)
    actors, organizations = _parse_projection_subjects(extracted_text)
    if not actors:
        twitter_profiles = simulation_manager.get_profiles(report.simulation_id, platform='twitter')
        reddit_profiles = simulation_manager.get_profiles(report.simulation_id, platform='reddit')
        actors = _profiles_to_actor_subjects(twitter_profiles or reddit_profiles)

    actions = SimulationRunner.get_all_actions(report.simulation_id)
    return {
        'report': report,
        'project': project,
        'simulation_state': simulation_state,
        'projection_meta': projection_meta,
        'extracted_text': extracted_text,
        'actors': _dedupe_subjects(actors),
        'organizations': _dedupe_subjects(organizations),
        'actions': actions,
        'graph_backend': graph_backend or simulation_state.graph_backend or project.graph_backend,
    }


def build_result_bundle(report: Any, *, graph_backend: str | None = None) -> dict[str, Any]:
    context = load_context(report, graph_backend=graph_backend)
    meta = context['projection_meta']
    project = context['project']
    simulation_state = context['simulation_state']
    actors = context['actors']
    organizations = context['organizations']
    actions = context['actions']
    subject_lookup = _build_subject_lookup(actors + organizations)
    world_id = str(meta.get('world_id') or project.project_id)
    scenario_id = str(meta.get('scenario_id') or report.simulation_id)
    run_id = str(report.report_id)

    runtime_evidence = []
    for action in actions:
        matched_ref = subject_lookup.get(_norm(action.agent_name))
        evidence_payload = {
            'run_id': run_id,
            'platform': action.platform,
            'round': action.round_num,
            'agent': action.agent_name,
            'action_type': action.action_type,
            'timestamp': _utc_iso(action.timestamp),
            'detail': _build_evidence_text(action),
        }
        runtime_evidence.append({
            'evidence_id': _stable_hex('ev', evidence_payload),
            'world_id': world_id,
            'scenario_id': scenario_id,
            'run_id': run_id,
            'evidence_type': 'agent_action',
            'source_type': 'mirofish_action_log',
            'actor_refs': [matched_ref] if matched_ref else [],
            'text': _build_evidence_text(action),
            'structured_payload': action.to_dict(),
            'timestamp': _utc_iso(action.timestamp),
            'confidence': 0.9 if getattr(action, 'success', True) else 0.4,
            'source_refs': [
                {'type': 'simulation', 'simulation_id': report.simulation_id},
                {'type': 'report', 'report_id': report.report_id},
                {'type': 'graph', 'graph_id': report.graph_id},
                {
                    'type': 'action_log',
                    'platform': action.platform,
                    'round_num': action.round_num,
                    'agent_id': action.agent_id,
                },
            ],
        })

    rumor_candidates = _derive_rumor_candidates(
        actions=actions,
        report=report,
        actors=actors,
        organizations=organizations,
        subject_lookup=subject_lookup,
    )
    emergent_events = _derive_emergent_events(
        actions=actions,
        report=report,
        actors=actors,
        organizations=organizations,
        subject_lookup=subject_lookup,
        project=project,
    )
    relationship_changes = _derive_relationship_changes(
        actions=actions,
        actors=actors,
        subject_lookup=subject_lookup,
    )

    _append_collection_evidence(
        runtime_evidence,
        rumor_candidates,
        world_id=world_id,
        scenario_id=scenario_id,
        run_id=run_id,
        report=report,
        collection_name='prediction_summary',
        evidence_type='rumor_signal',
        source_type='prediction_summary',
    )
    _append_collection_evidence(
        runtime_evidence,
        emergent_events,
        world_id=world_id,
        scenario_id=scenario_id,
        run_id=run_id,
        report=report,
        collection_name='emergent_event',
        evidence_type='world_event',
        source_type='emergent_event',
    )
    _append_collection_evidence(
        runtime_evidence,
        relationship_changes,
        world_id=world_id,
        scenario_id=scenario_id,
        run_id=run_id,
        report=report,
        collection_name='relationship_delta',
        evidence_type='relationship_change',
        source_type='relationship_delta',
    )
    candidate_deltas = _build_candidate_deltas(
        world_id=world_id,
        scenario_id=scenario_id,
        run_id=run_id,
        emergent_events=emergent_events,
        rumor_candidates=rumor_candidates,
        relationship_changes=relationship_changes,
        runtime_evidence=runtime_evidence,
    )

    prediction_summary = {
        'report_id': report.report_id,
        'report_status': getattr(report.status, 'value', report.status),
        'report_title': getattr(getattr(report, 'outline', None), 'title', None),
        'report_summary': getattr(getattr(report, 'outline', None), 'summary', None),
        'simulation_id': report.simulation_id,
        'project_id': project.project_id,
        'project_name': project.name,
        'graph_id': report.graph_id,
        'graph_backend': context['graph_backend'],
        'simulation_requirement': report.simulation_requirement,
        'analysis_summary': project.analysis_summary,
        'actors_count': len(actors),
        'organizations_count': len(organizations),
        'runtime_evidence_count': len(runtime_evidence),
        'platforms': sorted({action.platform for action in actions}),
        'rumors': rumor_candidates,
    }

    raw_payload = {
        'resolved_identifiers': {
            'world_id': world_id,
            'scenario_id': scenario_id,
            'run_id': run_id,
        },
        'project': {
            'project_id': project.project_id,
            'name': project.name,
            'graph_id': project.graph_id,
            'graph_backend': project.graph_backend,
            'analysis_summary': project.analysis_summary,
            'recommended_prepare_entity_types': project.recommended_prepare_entity_types,
        },
        'simulation_state': simulation_state.to_dict(),
        'report': report.to_dict(),
        'projection_meta': meta,
        'emergent_events': emergent_events,
        'relationship_changes': relationship_changes,
        'rumor_candidates': rumor_candidates,
        'actions_preview': [action.to_dict() for action in actions[:20]],
    }

    return {
        'schema_version': '1.1',
        'world_id': world_id,
        'scenario_id': scenario_id,
        'run_id': run_id,
        'generated_at': _utc_iso(report.completed_at),
        'world_version': meta.get('world_version') or project.updated_at,
        'projection_version': meta.get('schema_version'),
        'source_backend': f"mirofish:{context['graph_backend']}",
        'prediction_summary': prediction_summary,
        'actors': actors,
        'organizations': organizations,
        'emergent_events': emergent_events,
        'relationship_changes': relationship_changes,
        'rumor_candidates': rumor_candidates,
        'runtime_evidence': runtime_evidence,
        'candidate_deltas': candidate_deltas,
        'raw_payload': raw_payload,
    }


def _post_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    url = Config.MIROFISH_WRITEBACK_BASE_URL.rstrip('/') + '/api/mirofish/writeback/ingest'
    payload = json.dumps(bundle, ensure_ascii=False).encode('utf-8')
    request = urlrequest.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    with urlrequest.urlopen(request, timeout=Config.MIROFISH_WRITEBACK_TIMEOUT_SECONDS) as response:
        body = response.read().decode('utf-8')
        return {
            'url': url,
            'http_status': getattr(response, 'status', response.getcode()),
            'body': json.loads(body) if body else {},
        }


def _post_auto_promote(bundle: dict[str, Any]) -> dict[str, Any]:
    requests = _build_auto_promote_requests(bundle)
    if not requests:
        return {'enabled': True, 'skipped': True, 'reason': 'no_policy_eligible_candidates'}

    url = Config.MIROFISH_WRITEBACK_BASE_URL.rstrip('/') + '/api/mirofish/writeback/candidate-deltas/batch/auto-promote'
    responses: list[dict[str, Any]] = []
    for policy, items in requests:
        payload = json.dumps({'policy': policy, 'items': items}, ensure_ascii=False).encode('utf-8')
        request = urlrequest.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urlrequest.urlopen(request, timeout=Config.MIROFISH_WRITEBACK_TIMEOUT_SECONDS) as response:
            body = response.read().decode('utf-8')
            parsed = json.loads(body) if body else {}
            data = parsed.get('data') or {}
            responses.append({
                'policy': policy,
                'requested_count': len(items),
                'url': url,
                'http_status': getattr(response, 'status', response.getcode()),
                'body': parsed,
                'success_count': int(data.get('success_count') or 0),
                'failure_count': int(data.get('failure_count') or 0),
            })

    return {
        'enabled': True,
        'ok': all(item['http_status'] == 200 and item['failure_count'] == 0 for item in responses),
        'policy_count': len(responses),
        'attempted_candidate_count': sum(item['requested_count'] for item in responses),
        'success_count': sum(item['success_count'] for item in responses),
        'failure_count': sum(item['failure_count'] for item in responses),
        'responses': responses,
    }


def try_post_report_writeback(report: Any, *, graph_backend: str | None = None) -> dict[str, Any]:
    if not Config.MIROFISH_WRITEBACK_ENABLED:
        return {'enabled': False, 'skipped': True, 'reason': 'bridge_disabled'}

    try:
        bundle = build_result_bundle(report, graph_backend=graph_backend)
        posted = _post_bundle(bundle)
        auto_promote_result: dict[str, Any] | None = None
        if Config.MIROFISH_WRITEBACK_AUTO_PROMOTE_ENABLED:
            auto_promote_result = _post_auto_promote(bundle)
        logger.info(
            'Write-back bridge posted report=%s scenario=%s run=%s status=%s',
            report.report_id,
            bundle['scenario_id'],
            bundle['run_id'],
            posted['http_status'],
        )
        return {
            'enabled': True,
            'ok': bool(posted['http_status'] == 200 and (auto_promote_result is None or auto_promote_result.get('ok', False) or auto_promote_result.get('skipped'))),
            'url': posted['url'],
            'http_status': posted['http_status'],
            'world_id': bundle['world_id'],
            'scenario_id': bundle['scenario_id'],
            'run_id': bundle['run_id'],
            'runtime_evidence_count': len(bundle['runtime_evidence']),
            'candidate_deltas_count': len(bundle.get('candidate_deltas') or []),
            'subjects': {
                'actors': len(bundle['actors']),
                'organizations': len(bundle['organizations']),
            },
            'response': posted['body'],
            'auto_promote': auto_promote_result,
        }
    except (ValueError, OSError, json.JSONDecodeError, urlerror.URLError) as exc:
        logger.error('Write-back bridge failed for report %s: %s', getattr(report, 'report_id', 'unknown'), exc)
        return {
            'enabled': True,
            'ok': False,
            'error': str(exc),
            'report_id': getattr(report, 'report_id', None),
            'simulation_id': getattr(report, 'simulation_id', None),
        }