"""``bentrix.connect()`` with no arguments: find the application, pair, connect."""

import json
import os
import subprocess
import sys

import pytest

import bentrix
from bentrix import _discovery as d
from stub_server import TOKEN, Stub

SECRET = "the-secret"
# The application a test stands in for serves the catalogue this SDK was generated from.
CATALOGUE = bentrix.CATALOGUE_VERSION


def write_agents(folder, url, pid=None, secret=SECRET, catalogue=CATALOGUE):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / d.ENDPOINT_FILE).write_text(json.dumps({"url": url, "pid": os.getpid() if pid is None else pid, "catalogue_version": catalogue}))
    if secret is not None:
        (folder / d.SECRET_FILE).write_text(secret + "\n")


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_connect_finds_the_application_pairs_and_connects(tmp_path):
    with Stub(catalogue=CATALOGUE, pair_secret=SECRET) as stub:
        write_agents(tmp_path, stub.url)
        b = bentrix.connect(agents_folder=tmp_path, name="a script")
        # the generated module here is the empty catalogue, so the base client's call is what there is to try
        b.call("agent_whoami", {})
        path, headers, body = stub.pair_requests[0]
        assert path == "/mcp/pair"
        assert body == {"secret": SECRET, "client": {"name": "a script"}}
        assert "Authorization" not in headers, "the pairing request carries no bearer"
        assert stub.violations == [], "the token is the bearer on every later request"


def test_the_secret_travels_in_one_body_and_is_in_no_url_header_or_repr(tmp_path):
    with Stub(catalogue=CATALOGUE, pair_secret=SECRET) as stub:
        write_agents(tmp_path, stub.url)
        b = bentrix.connect(agents_folder=tmp_path)
        path, headers, _ = stub.pair_requests[0]
        assert SECRET not in path
        assert all(SECRET not in str(v) for v in headers.values())
        assert SECRET not in repr(b) and TOKEN not in repr(b)


def test_no_endpoint_file_says_that_bentrix_is_not_running_and_where_it_looked(tmp_path):
    with pytest.raises(bentrix.NotRunning) as e:
        bentrix.connect(agents_folder=tmp_path)
    assert "not running" in str(e.value) and str(tmp_path) in str(e.value)


def test_a_stale_endpoint_file_is_not_a_running_application(tmp_path):
    with Stub(catalogue=CATALOGUE) as stub:  # something is even listening: only the pid is stale
        write_agents(tmp_path, stub.url, pid=dead_pid())
        with pytest.raises(bentrix.NotRunning, match="stale"):
            bentrix.connect(agents_folder=tmp_path)
        assert stub.pair_requests == [], "a stale file is refused before anyone is asked anything"


def test_an_endpoint_that_does_not_answer_is_not_running(tmp_path):
    with Stub() as stub:
        url = stub.url
    write_agents(tmp_path, url)  # the stub has stopped: the pid is ours, the port is closed
    with pytest.raises(bentrix.NotRunning, match="does not answer"):
        bentrix.connect(agents_folder=tmp_path)


@pytest.mark.parametrize("content", ["not json", "[]", '{"url": "http://example.com/mcp", "pid": 1}', '{"pid": 1}'])
def test_an_endpoint_file_that_is_not_one_is_refused(tmp_path, content):
    (tmp_path / d.ENDPOINT_FILE).write_text(content)
    with pytest.raises(bentrix.NotRunning):
        bentrix.connect(agents_folder=tmp_path)


def test_a_missing_secret_file_is_a_clear_refusal(tmp_path):
    with Stub(catalogue=CATALOGUE) as stub:
        write_agents(tmp_path, stub.url, secret=None)
        with pytest.raises(bentrix.PairingRefused, match="local.secret"):
            bentrix.connect(agents_folder=tmp_path)


def test_a_wrong_secret_is_refused_without_repeating_it(tmp_path):
    with Stub(catalogue=CATALOGUE, pair_secret="another") as stub:
        write_agents(tmp_path, stub.url, secret="from-an-earlier-run")
        with pytest.raises(bentrix.PairingRefused) as e:
            bentrix.connect(agents_folder=tmp_path)
        assert "new secret at every start" in str(e.value)
        assert "from-an-earlier-run" not in str(e.value)


def test_a_rate_limited_pairing_says_to_wait(tmp_path):
    with Stub(catalogue=CATALOGUE, pair_status=429) as stub:
        write_agents(tmp_path, stub.url)
        with pytest.raises(bentrix.PairingRefused, match="wait"):
            bentrix.connect(agents_folder=tmp_path)


def test_given_a_url_and_a_token_nothing_is_read_or_asked(tmp_path):
    with Stub(catalogue=CATALOGUE) as stub:
        b = bentrix.connect(stub.url, TOKEN, agents_folder=tmp_path / "does-not-exist")
        b.call("agent_whoami", {})
        assert stub.pair_requests == []


def test_given_only_a_url_it_pairs_with_that_endpoint(tmp_path):
    with Stub(catalogue=CATALOGUE, pair_secret=SECRET) as stub:
        write_agents(tmp_path, "http://127.0.0.1:1/unused")
        bentrix.connect(stub.url, agents_folder=tmp_path)
        assert len(stub.pair_requests) == 1


def test_the_default_name_is_the_environment_then_the_script(monkeypatch):
    monkeypatch.setenv("BENTRIX_AGENT_NAME", "  my agent ")
    assert d.default_name() == "my agent"
    monkeypatch.delenv("BENTRIX_AGENT_NAME")
    monkeypatch.setattr(sys, "argv", ["/x/y/analyse.py"])
    assert d.default_name() == "python sdk (analyse.py)"
    monkeypatch.setattr(sys, "argv", [""])
    assert d.default_name() == "python sdk"


def test_the_agents_folder_mirrors_the_applications_own_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert d.agents_dir() == tmp_path / "local" / "BENTRIX" / "agents", "LOCALAPPDATA wins wherever it is set"
    monkeypatch.setenv("LOCALAPPDATA", "")
    assert d.agents_dir() == tmp_path / "home" / ".local" / "share" / "bentrix" / "agents", "an empty one is as good as none"
    monkeypatch.delenv("LOCALAPPDATA")
    assert d.agents_dir() == tmp_path / "home" / ".local" / "share" / "bentrix" / "agents"
    monkeypatch.setenv("HOME", "")
    assert str(d.agents_dir()) == "bentrix-agents", "the application's last resort is relative, and so is this"


def test_a_live_pid_is_alive_and_a_finished_one_is_not():
    assert d.pid_alive(os.getpid())
    assert not d.pid_alive(dead_pid())
    assert not d.pid_alive(0) and not d.pid_alive(-5)


def test_an_answer_with_no_token_is_not_a_pairing(tmp_path):
    with Stub(catalogue=CATALOGUE, pair_secret=SECRET, pair_answer={"token": ""}) as stub:
        write_agents(tmp_path, stub.url)
        with pytest.raises(bentrix.PairingRefused, match="no token"):
            bentrix.connect(agents_folder=tmp_path)
