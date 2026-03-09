import csv

from app.services.oasis_profile_generator import OasisAgentProfile, OasisProfileGenerator


def test_build_twitter_csv_rows_uses_oasis_schema_and_combines_persona():
    profile = OasisAgentProfile(
        user_id=10,
        user_name='captain_mira',
        name='Captain Mira',
        bio='Harbor captain monitoring the docks.',
        persona='Calm, procedural, and trusted in crises.',
    )

    rows = OasisProfileGenerator._build_twitter_csv_rows([profile])

    assert rows == [{
        'user_id': 0,
        'name': 'Captain Mira',
        'username': 'captain_mira',
        'user_char': 'Harbor captain monitoring the docks. Calm, procedural, and trusted in crises.',
        'description': 'Harbor captain monitoring the docks.',
    }]


def test_save_twitter_csv_writes_user_char_column(tmp_path):
    generator = OasisProfileGenerator.__new__(OasisProfileGenerator)
    profile = OasisAgentProfile(
        user_id=5,
        user_name='dockworker_len',
        name='Dockworker Len',
        bio='Dockworker watching for labor trouble.',
        persona='Quick to spread rumors if management stays silent.',
    )
    path = tmp_path / 'twitter_profiles.csv'

    generator._save_twitter_csv([profile], str(path))

    with open(path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    assert list(rows[0].keys()) == ['user_id', 'name', 'username', 'user_char', 'description']
    assert rows[0]['username'] == 'dockworker_len'
    assert 'Quick to spread rumors' in rows[0]['user_char']