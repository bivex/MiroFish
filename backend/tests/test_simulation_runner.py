import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_simulation_runner_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    config_module = types.ModuleType("app.config")

    class Config:
        @classmethod
        def get_graph_backend(cls):
            return "cognee"

    config_module.Config = Config
    sys.modules["app.config"] = config_module

    logger_module = types.ModuleType("app.utils.logger")
    logger_module.get_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    sys.modules["app.utils.logger"] = logger_module

    factory_module = types.ModuleType("app.services.graph_backend_factory")

    class _MemoryManager:
        @staticmethod
        def stop_all():
            return None

        @staticmethod
        def get_updater(simulation_id):
            return None

        @staticmethod
        def stop_updater(simulation_id):
            return None

    factory_module.get_graph_memory_manager_class = lambda graph_backend=None: _MemoryManager
    sys.modules["app.services.graph_backend_factory"] = factory_module

    ipc_module = types.ModuleType("app.services.simulation_ipc")

    class SimulationIPCClient:
        def __init__(self, sim_dir):
            self.sim_dir = sim_dir

        def check_env_alive(self):
            return False

    class CommandType:
        pass

    class IPCResponse:
        pass

    ipc_module.SimulationIPCClient = SimulationIPCClient
    ipc_module.CommandType = CommandType
    ipc_module.IPCResponse = IPCResponse
    sys.modules["app.services.simulation_ipc"] = ipc_module

    module_path = ROOT / "app" / "services" / "simulation_runner.py"
    spec = importlib.util.spec_from_file_location("app.services.simulation_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.simulation_runner"] = module
    spec.loader.exec_module(module)
    return module


def test_get_run_state_reconciles_dead_running_process_to_stopped(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_stale"
    sim_dir.mkdir()

    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_stale",
        "project_id": "proj_1",
        "graph_id": "graph_1",
        "status": "running",
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")
    (sim_dir / "run_state.json").write_text(json.dumps({
        "simulation_id": "sim_stale",
        "runner_status": "running",
        "current_round": 3,
        "total_rounds": 24,
        "twitter_running": True,
        "reddit_running": False,
        "process_pid": 999999,
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")

    state = runner.get_run_state("sim_stale")

    assert state is not None
    assert state.runner_status == module.RunnerStatus.STOPPED
    assert state.process_pid is None
    assert state.twitter_running is False

    synced_state = json.loads((sim_dir / "state.json").read_text(encoding="utf-8"))
    synced_run_state = json.loads((sim_dir / "run_state.json").read_text(encoding="utf-8"))

    assert synced_state["status"] == "stopped"
    assert synced_run_state["runner_status"] == "stopped"
    assert synced_run_state["process_pid"] is None


def test_get_run_state_marks_dead_process_completed_when_simulation_end_exists(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_completed"
    twitter_dir = sim_dir / "twitter"
    twitter_dir.mkdir(parents=True)

    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_completed",
        "project_id": "proj_1",
        "graph_id": "graph_1",
        "status": "running",
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")
    (sim_dir / "run_state.json").write_text(json.dumps({
        "simulation_id": "sim_completed",
        "runner_status": "running",
        "current_round": 0,
        "total_rounds": 0,
        "twitter_running": True,
        "reddit_running": False,
        "process_pid": 999999,
        "updated_at": "2026-03-09T00:00:00",
        "twitter_actions_count": 0,
        "reddit_actions_count": 0,
    }), encoding="utf-8")
    (twitter_dir / "actions.jsonl").write_text("\n".join([
        json.dumps({
            "platform": "twitter",
            "round": 2,
            "agent_id": 1,
            "agent_name": "Captain Mira",
            "action_type": "create_post",
            "action_args": {"content": "Harbor update"},
            "success": True,
        }),
        json.dumps({
            "event_type": "round_end",
            "platform": "twitter",
            "round": 2,
            "simulated_hours": 2,
        }),
        json.dumps({
            "event_type": "simulation_end",
            "platform": "twitter",
            "total_rounds": 24,
            "total_actions": 1,
        }),
    ]), encoding="utf-8")

    state = runner.get_run_state("sim_completed")

    assert state is not None
    assert state.runner_status == module.RunnerStatus.COMPLETED
    assert state.twitter_completed is True
    assert state.twitter_actions_count == 1
    assert state.current_round == 2
    assert state.total_rounds == 24

    synced_state = json.loads((sim_dir / "state.json").read_text(encoding="utf-8"))
    synced_run_state = json.loads((sim_dir / "run_state.json").read_text(encoding="utf-8"))

    assert synced_state["status"] == "completed"
    assert synced_run_state["runner_status"] == "completed"
    assert synced_run_state["twitter_actions_count"] == 1
    assert synced_run_state["current_round"] == 2


def test_get_run_state_reconciles_untracked_alive_process_after_restart(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_orphaned"
    sim_dir.mkdir()

    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_orphaned",
        "project_id": "proj_1",
        "graph_id": "graph_1",
        "status": "running",
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")
    (sim_dir / "run_state.json").write_text(json.dumps({
        "simulation_id": "sim_orphaned",
        "runner_status": "running",
        "current_round": 0,
        "total_rounds": 24,
        "twitter_running": True,
        "reddit_running": False,
        "process_pid": 424242,
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")

    terminated = []
    runner._is_pid_alive = classmethod(lambda cls, pid: False if terminated else True)
    runner._pid_matches_simulation_process = classmethod(lambda cls, pid, simulation_id: True)
    runner._terminate_orphaned_process = classmethod(
        lambda cls, pid, simulation_id, timeout=10: terminated.append((pid, simulation_id)) or True
    )

    state = runner.get_run_state("sim_orphaned")

    assert state is not None
    assert state.runner_status == module.RunnerStatus.STOPPED
    assert state.process_pid is None
    assert terminated == [(424242, "sim_orphaned")]

    synced_state = json.loads((sim_dir / "state.json").read_text(encoding="utf-8"))
    synced_run_state = json.loads((sim_dir / "run_state.json").read_text(encoding="utf-8"))

    assert synced_state["status"] == "stopped"
    assert synced_run_state["runner_status"] == "stopped"
    assert synced_run_state["process_pid"] is None


def test_get_run_state_cleans_completed_process_when_env_already_stopped(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_completed_orphan"
    sim_dir.mkdir()

    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_completed_orphan",
        "project_id": "proj_1",
        "graph_id": "graph_1",
        "status": "completed",
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")
    (sim_dir / "run_state.json").write_text(json.dumps({
        "simulation_id": "sim_completed_orphan",
        "runner_status": "completed",
        "current_round": 24,
        "total_rounds": 24,
        "twitter_running": False,
        "reddit_running": False,
        "process_pid": 313131,
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")
    (sim_dir / "env_status.json").write_text(json.dumps({
        "status": "stopped",
        "timestamp": "2026-03-09T00:01:00",
    }), encoding="utf-8")

    terminated = []
    runner._is_pid_alive = classmethod(lambda cls, pid: False if terminated else True)
    runner._pid_matches_simulation_process = classmethod(lambda cls, pid, simulation_id: True)
    runner._terminate_orphaned_process = classmethod(
        lambda cls, pid, simulation_id, timeout=10: terminated.append((pid, simulation_id)) or True
    )

    state = runner.get_run_state("sim_completed_orphan")

    assert state is not None
    assert state.runner_status == module.RunnerStatus.COMPLETED
    assert state.process_pid is None
    assert terminated == [(313131, "sim_completed_orphan")]

    synced_run_state = json.loads((sim_dir / "run_state.json").read_text(encoding="utf-8"))
    assert synced_run_state["runner_status"] == "completed"
    assert synced_run_state["process_pid"] is None


def test_close_simulation_env_reconciles_stuck_process_after_shutdown(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()
    runner._expected_process_exits.clear()

    sim_dir = tmp_path / "sim_close_env"
    sim_dir.mkdir()

    (sim_dir / "run_state.json").write_text(json.dumps({
        "simulation_id": "sim_close_env",
        "runner_status": "completed",
        "current_round": 24,
        "total_rounds": 24,
        "twitter_running": False,
        "reddit_running": False,
        "process_pid": 515151,
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")
    (sim_dir / "env_status.json").write_text(json.dumps({
        "status": "alive",
        "timestamp": "2026-03-09T00:00:00",
    }), encoding="utf-8")

    class FakeIPCClient:
        def __init__(self, sim_dir_path):
            self.sim_dir = Path(sim_dir_path)

        def check_env_alive(self):
            payload = json.loads((self.sim_dir / "env_status.json").read_text(encoding="utf-8"))
            return payload.get("status") == "alive"

        def send_close_env(self, timeout=30.0):
            (self.sim_dir / "env_status.json").write_text(json.dumps({
                "status": "stopped",
                "timestamp": "2026-03-09T00:00:01",
            }), encoding="utf-8")
            return types.SimpleNamespace(
                status=types.SimpleNamespace(value="completed"),
                result={"message": "环境即将关闭"},
                timestamp="2026-03-09T00:00:01",
            )

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.alive = True

        def poll(self):
            return None if self.alive else 0

    terminated = []
    module.SimulationIPCClient = FakeIPCClient
    runner._is_pid_alive = classmethod(lambda cls, pid: False if terminated else True)
    runner._pid_matches_simulation_process = classmethod(lambda cls, pid, simulation_id: True)
    runner._terminate_process = classmethod(
        lambda cls, process, simulation_id, timeout=10: terminated.append((process.pid, simulation_id)) or setattr(process, "alive", False)
    )
    runner._processes["sim_close_env"] = FakeProcess(515151)

    result = runner.close_simulation_env("sim_close_env", timeout=1.0)

    assert result["success"] is True
    assert terminated == [(515151, "sim_close_env")]

    synced_run_state = json.loads((sim_dir / "run_state.json").read_text(encoding="utf-8"))
    assert synced_run_state["process_pid"] is None


def test_start_simulation_rejects_restart_while_process_still_alive(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_waiting"
    sim_dir.mkdir()

    (sim_dir / "simulation_config.json").write_text(json.dumps({
        "time_config": {
            "total_simulation_hours": 24,
            "minutes_per_round": 60,
        }
    }), encoding="utf-8")
    (sim_dir / "run_state.json").write_text(json.dumps({
        "simulation_id": "sim_waiting",
        "runner_status": "completed",
        "current_round": 24,
        "total_rounds": 24,
        "twitter_running": False,
        "reddit_running": False,
        "process_pid": 616161,
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    runner._processes["sim_waiting"] = FakeProcess(616161)
    runner._pid_matches_simulation_process = classmethod(lambda cls, pid, simulation_id: True)

    try:
        runner.start_simulation("sim_waiting")
    except ValueError as exc:
        assert "/close-env" in str(exc)
    else:
        raise AssertionError("expected start_simulation to reject overlapping restart")


def test_start_simulation_recovers_stale_dead_process_after_restart(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner.SCRIPTS_DIR = str(tmp_path / "scripts")
    runner._run_states.clear()
    runner._processes.clear()
    runner._action_queues.clear()
    runner._monitor_threads.clear()
    runner._stdout_files.clear()
    runner._stderr_files.clear()

    sim_dir = tmp_path / "sim_restartable"
    sim_dir.mkdir()
    scripts_dir = Path(runner.SCRIPTS_DIR)
    scripts_dir.mkdir()
    (scripts_dir / "run_twitter_simulation.py").write_text("# twitter\n", encoding="utf-8")

    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_restartable",
        "enable_twitter": True,
        "enable_reddit": False,
        "status": "running",
    }), encoding="utf-8")
    (sim_dir / "simulation_config.json").write_text(json.dumps({
        "time_config": {
            "total_simulation_hours": 24,
            "minutes_per_round": 60,
        }
    }), encoding="utf-8")
    (sim_dir / "run_state.json").write_text(json.dumps({
        "simulation_id": "sim_restartable",
        "runner_status": "running",
        "current_round": 5,
        "total_rounds": 24,
        "twitter_running": True,
        "reddit_running": False,
        "process_pid": 717171,
        "updated_at": "2026-03-09T00:00:00",
    }), encoding="utf-8")

    launched = {}

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            launched["cmd"] = cmd
            launched["cwd"] = kwargs.get("cwd")
            self.pid = 818181

        def poll(self):
            return None

    class FakeThread:
        def __init__(self, target=None, args=None, daemon=None):
            self.target = target
            self.args = args or ()

        def start(self):
            launched["monitor_started"] = True

    module.subprocess.Popen = FakeProcess
    module.threading.Thread = FakeThread
    runner._is_pid_alive = classmethod(lambda cls, pid: False)

    state = runner.start_simulation("sim_restartable", platform="parallel")

    assert Path(launched["cmd"][1]).name == "run_twitter_simulation.py"
    assert launched["cwd"] == str(sim_dir)
    assert launched["monitor_started"] is True
    assert state.runner_status == module.RunnerStatus.RUNNING
    assert state.process_pid == 818181
    assert state.twitter_running is True
    assert state.reddit_running is False

    synced_state = json.loads((sim_dir / "state.json").read_text(encoding="utf-8"))
    synced_run_state = json.loads((sim_dir / "run_state.json").read_text(encoding="utf-8"))

    assert synced_state["status"] == "stopped"
    assert synced_run_state["runner_status"] == "running"
    assert synced_run_state["process_pid"] == 818181


def test_read_action_log_marks_platform_completed_without_finishing_live_runner(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_live"
    twitter_dir = sim_dir / "twitter"
    twitter_dir.mkdir(parents=True)
    log_path = twitter_dir / "actions.jsonl"
    log_path.write_text(json.dumps({
        "event_type": "simulation_end",
        "platform": "twitter",
        "total_rounds": 24,
        "total_actions": 5,
    }) + "\n", encoding="utf-8")

    messages = []
    module.logger = types.SimpleNamespace(
        info=lambda msg: messages.append(msg),
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    state = module.SimulationRunState(
        simulation_id="sim_live",
        runner_status=module.RunnerStatus.RUNNING,
        process_pid=12345,
        twitter_running=True,
        reddit_running=False,
    )

    runner._read_action_log(str(log_path), 0, state, "twitter")
    runner._read_action_log(str(log_path), 0, state, "twitter")

    assert state.twitter_completed is True
    assert state.twitter_running is False
    assert state.runner_status == module.RunnerStatus.RUNNING
    assert state.completed_at is None
    assert sum("Twitter simulation completed" in msg for msg in messages) == 1
    assert sum("All enabled platform simulations completed" in msg for msg in messages) == 1


def test_get_public_run_state_dict_marks_waiting_completed_for_read_path(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_waiting_completed"
    twitter_dir = sim_dir / "twitter"
    twitter_dir.mkdir(parents=True)
    (twitter_dir / "actions.jsonl").write_text(json.dumps({
        "event_type": "simulation_end",
        "platform": "twitter",
        "total_rounds": 12,
        "total_actions": 18,
    }) + "\n", encoding="utf-8")

    state = module.SimulationRunState(
        simulation_id="sim_waiting_completed",
        runner_status=module.RunnerStatus.RUNNING,
        process_pid=424242,
        current_round=12,
        total_rounds=12,
        twitter_running=False,
        reddit_running=False,
        twitter_completed=True,
        twitter_actions_count=18,
    )

    payload = runner.get_public_run_state_dict(state)

    assert payload is not None
    assert payload["runner_status"] == "completed"
    assert payload["completed_at"] == state.updated_at
    assert state.runner_status == module.RunnerStatus.RUNNING
    assert state.process_pid == 424242


def test_get_public_run_state_dict_keeps_running_while_platform_loop_still_active(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()

    sim_dir = tmp_path / "sim_still_running"
    twitter_dir = sim_dir / "twitter"
    twitter_dir.mkdir(parents=True)
    (twitter_dir / "actions.jsonl").write_text("", encoding="utf-8")

    state = module.SimulationRunState(
        simulation_id="sim_still_running",
        runner_status=module.RunnerStatus.RUNNING,
        process_pid=515151,
        current_round=3,
        total_rounds=12,
        twitter_running=True,
        reddit_running=False,
        twitter_completed=False,
        twitter_actions_count=4,
    )

    payload = runner.get_public_run_state_dict(state)

    assert payload is not None
    assert payload["runner_status"] == "running"
    assert payload["completed_at"] is None


def test_start_simulation_coerces_parallel_to_single_enabled_platform(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner.SCRIPTS_DIR = str(tmp_path / "scripts")
    runner._run_states.clear()
    runner._processes.clear()
    runner._action_queues.clear()
    runner._monitor_threads.clear()
    runner._stdout_files.clear()
    runner._stderr_files.clear()

    sim_dir = tmp_path / "sim_twitter_only"
    sim_dir.mkdir()
    scripts_dir = Path(runner.SCRIPTS_DIR)
    scripts_dir.mkdir()
    (scripts_dir / "run_twitter_simulation.py").write_text("# twitter\n", encoding="utf-8")
    (scripts_dir / "run_reddit_simulation.py").write_text("# reddit\n", encoding="utf-8")
    (scripts_dir / "run_parallel_simulation.py").write_text("# parallel\n", encoding="utf-8")

    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_twitter_only",
        "enable_twitter": True,
        "enable_reddit": False,
        "status": "ready",
    }), encoding="utf-8")
    (sim_dir / "simulation_config.json").write_text(json.dumps({
        "time_config": {
            "total_simulation_hours": 24,
            "minutes_per_round": 60,
        }
    }), encoding="utf-8")

    launched = {}

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            launched["cmd"] = cmd
            launched["cwd"] = kwargs.get("cwd")
            self.pid = 737373

        def poll(self):
            return None

    class FakeThread:
        def __init__(self, target=None, args=None, daemon=None):
            self.target = target
            self.args = args or ()

        def start(self):
            launched["monitor_started"] = True

    module.subprocess.Popen = FakeProcess
    module.threading.Thread = FakeThread

    state = runner.start_simulation("sim_twitter_only", platform="parallel")

    assert Path(launched["cmd"][1]).name == "run_twitter_simulation.py"
    assert state.twitter_running is True
    assert state.reddit_running is False


def test_get_env_status_detail_falls_back_to_enabled_platform_flags(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)

    sim_dir = tmp_path / 'sim_env'
    sim_dir.mkdir()
    (sim_dir / 'state.json').write_text(json.dumps({
        'simulation_id': 'sim_env',
        'enable_twitter': True,
        'enable_reddit': False,
    }), encoding='utf-8')
    (sim_dir / 'env_status.json').write_text(json.dumps({
        'status': 'alive',
        'timestamp': '2026-03-09T00:00:00',
    }), encoding='utf-8')

    status = runner.get_env_status_detail('sim_env')

    assert status['status'] == 'alive'
    assert status['twitter_available'] is True
    assert status['reddit_available'] is False


def test_monitor_simulation_marks_close_env_as_completed_when_loop_already_finished(tmp_path):
    module = load_simulation_runner_module()
    runner = module.SimulationRunner
    runner.RUN_STATE_DIR = str(tmp_path)
    runner._run_states.clear()
    runner._processes.clear()
    runner._action_queues.clear()
    runner._monitor_threads.clear()
    runner._stdout_files.clear()
    runner._stderr_files.clear()
    runner._expected_process_exits.clear()

    sim_dir = tmp_path / 'sim_close_completed'
    twitter_dir = sim_dir / 'twitter'
    twitter_dir.mkdir(parents=True)
    (sim_dir / 'state.json').write_text(json.dumps({
        'simulation_id': 'sim_close_completed',
        'status': 'running',
        'enable_twitter': True,
        'enable_reddit': False,
    }), encoding='utf-8')
    (twitter_dir / 'actions.jsonl').write_text(json.dumps({
        'event_type': 'simulation_end',
        'platform': 'twitter',
        'total_rounds': 3,
        'total_actions': 0,
    }) + '\n', encoding='utf-8')

    state = module.SimulationRunState(
        simulation_id='sim_close_completed',
        runner_status=module.RunnerStatus.RUNNING,
        process_pid=424242,
        twitter_running=True,
        reddit_running=False,
    )
    runner._run_states['sim_close_completed'] = state
    runner._expected_process_exits['sim_close_completed'] = 'close_env'

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self._poll_count = 0

        def poll(self):
            self._poll_count += 1
            if self._poll_count == 1:
                return None
            self.returncode = -15
            return self.returncode

    runner._processes['sim_close_completed'] = FakeProcess()
    module.time.sleep = lambda *_args, **_kwargs: None

    runner._monitor_simulation('sim_close_completed')

    assert state.runner_status == module.RunnerStatus.COMPLETED
    assert state.twitter_completed is True
    assert state.process_pid is None

    synced_state = json.loads((sim_dir / 'state.json').read_text(encoding='utf-8'))
    synced_run_state = json.loads((sim_dir / 'run_state.json').read_text(encoding='utf-8'))

    assert synced_state['status'] == 'completed'
    assert synced_run_state['runner_status'] == 'completed'