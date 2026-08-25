"""Test dell'app interattiva.

L'app e' la cosa che si apre davanti a qualcuno, quindi il modo peggiore di scoprire che e'
rotta e' durante una presentazione. Qui si verifica che i tre stadi rispondano, che il
percorso senza verita' resti senza verita', e che le due regole che l'app deve *far vedere*
valgano davvero: la mistura resta una CDF valida quando le si spegne una componente, e la
diagnostica che usa la verita' sparisce quando la verita' non c'e'.

Non si verifica l'aspetto: quello va guardato.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

fastapi_testclient = pytest.importorskip("fastapi.testclient")
server = pytest.importorskip("server")


@pytest.fixture(scope="module")
def cli():
    return fastapi_testclient.TestClient(server.app)


BIMODALE = [
    {"type": "gaussian", "params": {"mean": -2.0, "std": 0.5}, "weight": 1.0},
    {"type": "gaussian", "params": {"mean": 2.0, "std": 0.8}, "weight": 1.0},
]
BASE = {"components": BIMODALE, "n": 300, "seed": 0, "kernel": "logistic",
        "strategy": "lscv", "epochs": 200, "net_seed": 0}


def _addestra(cli, **cfg):
    """Un giro di addestramento completo; restituisce (hello, ultima istantanea)."""
    with cli.websocket_connect("/ws/train") as ws:
        ws.send_text(json.dumps({"cmd": "start", "config": {**BASE, **cfg}}))
        hello = ultima = None
        while True:
            m = ws.receive_json()
            if m["type"] == "hello":
                hello = m
            elif m["type"] == "snapshot":
                ultima = m
            elif m["type"] == "done":
                break
            elif m["type"] == "error":
                pytest.fail(f"il server ha risposto errore: {m['message']}")
    return hello, ultima


# ------------------------------------------------------------------ le risorse di base

def test_la_pagina_e_gli_asset_rispondono(cli):
    assert cli.get("/").status_code == 200
    assert cli.get("/static/app.js").status_code == 200
    assert cli.get("/static/style.css").status_code == 200


def test_il_registro_espone_le_scelte_consegnate(cli):
    reg = cli.get("/api/registries").json()
    assert reg["strategies"][0] == "lscv"          # l'iniziale e' la prima della lista
    assert "sqrt_n" in reg["strategies"]           # la forma richiesta dall'esercizio
    assert "logistic" in reg["kernels"]


# ------------------------------------------------------------------ stadio 2

def test_la_stima_di_parzen_riporta_h1_e_la_diagnostica(cli):
    d = cli.post("/api/parzen", json=BASE).json()
    assert d["n"] == 300
    assert d["h_summary"]["h1"] == pytest.approx(d["h"] * np.sqrt(300))
    g = d["diagnostics"]
    assert g["lscv_score"] is not None and g["loo_loglik"] is not None
    assert d["ks_vs_truth"] is not None            # qui la verita' c'e'


def test_il_ks_contro_l_ecdf_migliora_mentre_l_errore_vero_peggiora(cli):
    """La trappola che l'app esiste per mostrare (D-14). Se smettesse di valere, la
    tessera-trappola direbbe una cosa falsa."""
    stretta = cli.post("/api/parzen", json={**BASE, "strategy": "manual", "h_manual": 0.02}).json()
    giusta = cli.post("/api/parzen", json={**BASE, "strategy": "lscv"}).json()
    assert stretta["ks_vs_ecdf"] < giusta["ks_vs_ecdf"]      # sembra migliore
    assert stretta["ks_vs_truth"] > giusta["ks_vs_truth"]    # ed e' peggiore


def test_i_selettori_classici_sovralisciano(cli):
    silverman = cli.post("/api/parzen", json={**BASE, "strategy": "silverman"}).json()
    lscv = cli.post("/api/parzen", json={**BASE, "strategy": "lscv"}).json()
    assert silverman["h"] > lscv["h"]
    assert silverman["ks_vs_truth"] > lscv["ks_vs_truth"]


def test_la_cross_validazione_ha_un_tetto(cli):
    r = cli.post("/api/parzen", json={**BASE, "n": 5000, "strategy": "lscv"})
    assert r.status_code == 400 and "capped" in r.json()["error"]


# ------------------------------------------------------------------ stadio 3

def test_la_mistura_esce_valida_dall_addestramento(cli):
    hello, ultima = _addestra(cli, estimator="mixture", n_components=12)
    assert hello["estimator"] == "mixture"
    assert hello["layer_sizes"] == [1, 12, 1]
    assert ultima["violations"] == 0
    assert ultima["mass"] == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("seme", [0, 1, 2])
def test_solo_la_mistura_garantisce_la_massa(cli, seme):
    """Il confronto che l'app deve rendere visibile.

    Non si verifica che l'MLP violi la monotonia, perche' non e' vero che la viola sempre:
    dipende dal seme, ed e' esattamente il punto (assente per fortuna, non per garanzia).
    Cio' che separa i due casi in modo deterministico e' la massa. L'MLP non arriva alle
    code, quindi ne perde qualche punto percentuale a ogni seme; la mistura vale 1 esatto
    perche' i suoi limiti sono 0 e 1 per costruzione.
    """
    _, mix = _addestra(cli, estimator="mixture", epochs=300, net_seed=seme)
    _, mlp = _addestra(cli, estimator="mlp", hidden=[32], epochs=300, net_seed=seme)
    assert mix["violations"] == 0
    assert mix["mass"] == pytest.approx(1.0, abs=1e-4)
    assert abs(mlp["mass"] - 1.0) > 1e-2


def test_la_rettifica_non_abbassa_le_violazioni_dichiarate(cli):
    """Sono misurate sulla curva grezza: la toppa nasconde il sintomo, non lo cura."""
    _, senza = _addestra(cli, estimator="mlp", hidden=[32], rectify=False)
    _, con = _addestra(cli, estimator="mlp", hidden=[32], rectify=True)
    assert con["violations"] == senza["violations"]
    assert con["mass"] == pytest.approx(1.0, abs=1e-6)     # la massa si', quella la impone


def test_spegnere_una_componente_lascia_una_cdf_valida(cli):
    """Il teorema di validita', reso cliccabile."""
    with cli.websocket_connect("/ws/train") as ws:
        ws.send_text(json.dumps({"cmd": "start", "config": {**BASE, "estimator": "mixture"}}))
        while ws.receive_json()["type"] != "done":
            pass
        ws.send_text(json.dumps({"cmd": "prune", "l": 1, "o": 0, "i": 5}))
        dopo = ws.receive_json()
        assert dopo["pruned"] == [[1, 0, 5]]
        assert dopo["violations"] == 0
        assert dopo["mass"] == pytest.approx(1.0, abs=2e-3)

        # l'arco d'ingresso invece renderebbe l'unita' costante a 1/2: si rifiuta
        ws.send_text(json.dumps({"cmd": "prune", "l": 0, "o": 5, "i": 0}))
        rifiuto = ws.receive_json()
        assert rifiuto["type"] == "error" and "outgoing" in rifiuto["message"]


def test_il_maestro_leave_one_out_e_l_iniziale(cli):
    hello, _ = _addestra(cli, estimator="mixture")
    y = hello["target_points"]["y"]
    # le etichette LOO tolgono K(0) = 1/2 dal punto stesso, quindi si allontanano da 1/2
    hello_auto, _ = _addestra(cli, estimator="mixture", target="parzen")
    assert y != hello_auto["target_points"]["y"]


# ------------------------------------------------------------------ il campione esterno

def test_un_file_di_numeri_attraversa_tutta_la_pipeline(cli):
    x = np.concatenate([np.random.default_rng(1).normal(-2, 0.4, 150),
                        np.random.default_rng(2).normal(2.5, 0.7, 150)])
    testo = "\n".join(f"{v:.6f}" for v in x)
    letto = cli.post("/api/samples", json={"text": testo}).json()
    assert letto["n"] == 300

    d = cli.post("/api/parzen", json={"samples": letto["samples"], "strategy": "lscv"}).json()
    assert d["truth_cdf"] is None and d["truth_pdf"] is None
    assert d["ks_vs_truth"] is None                    # niente verita' da confrontare
    assert d["diagnostics"]["lscv_score"] is not None  # la diagnostica senza verita' resta

    _, ultima = _addestra(cli, samples=letto["samples"], components=None, estimator="mixture")
    assert ultima["ks_truth"] is None
    assert ultima["violations"] == 0
    assert ultima["mass"] == pytest.approx(1.0, abs=2e-3)


def test_il_dominio_dai_quantili_veri_non_esiste_su_un_file(cli):
    testo = "\n".join(str(v) for v in np.random.default_rng(0).normal(size=50))
    campioni = cli.post("/api/samples", json={"text": testo}).json()["samples"]
    r = cli.post("/api/parzen", json={"samples": campioni, "domain": "truth"})
    assert r.status_code == 400 and "no distribution" in r.json()["error"]


@pytest.mark.parametrize("testo, atteso", [
    ("\n".join(str(i / 10) for i in range(25)), 25),          # una colonna
    ("x\n" + "\n".join(str(i / 10) for i in range(25)), 25),  # con intestazione
    (",".join(str(i / 10) for i in range(25)), 25),           # vettore su una riga
])
def test_i_formati_leggibili_si_leggono(cli, testo, atteso):
    assert cli.post("/api/samples", json={"text": testo}).json()["n"] == atteso


@pytest.mark.parametrize("testo", [
    "\n".join(f"{i / 10}".replace(".", ",") for i in range(25)),   # virgola decimale
    "\n".join(f"{i},{i / 10}" for i in range(25)),                 # due colonne
])
def test_i_formati_ambigui_si_rifiutano_invece_di_indovinare(cli, testo):
    """Il caso pericoloso: entrambe le letture producono numeri, e una e' inventata."""
    r = cli.post("/api/samples", json={"text": testo})
    assert r.status_code == 400
    assert "column" in r.json()["error"] or "decimal" in r.json()["error"]


def test_una_volta_indicata_la_lettura_il_file_passa(cli):
    virgola = "\n".join(f"{i / 10}".replace(".", ",") for i in range(25))
    d = cli.post("/api/samples", json={"text": virgola, "decimal": "comma"}).json()
    assert d["n"] == 25 and d["summary"]["max"] == pytest.approx(2.4)

    due = "\n".join(f"{i},{i / 10}" for i in range(25))
    d = cli.post("/api/samples", json={"text": due, "column": 2}).json()
    assert d["n"] == 25 and d["summary"]["max"] == pytest.approx(2.4)
