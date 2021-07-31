import json
import numpy as np
from pathlib import Path

def retrieve_ED_ME_coupled_sqlite_single(a, b, pol_vec, rme_only, con):
    assert not rme_only is None, "rme_only cannot be None"
    table = 'ED_ME_coupled'
    with con:
        cur = con.cursor()
        query = generate_ED_ME_coupled_query(a,b,pol_vec,rme_only)
        cur.execute(f"select value_real, value_imag from {table} WHERE {query}")
        values = cur.fetchall()
        if values:
            values = values[0]
            values = values[0] + 1j*values[1]
        else:
            values = complex(0)
    return values

def generate_ED_ME_coupled_query(a,b,pol_vec,rme_only):
    P1 = a.P if a.P is not None else 0
    P2 = b.P if b.P is not None else 0

    query = f"J₁ = {a.J} AND F1₁ = {a.F1} AND F₁ = {int(a.F)} AND mF₁ = {int(a.mF)} AND I1₁ = {a.I1} AND I2₁ = {a.I2} AND P₁ = {P1} AND Ω₁ = {int(a.Omega)} AND state₁ = '{a.electronic_state}' "
    query += f"AND J₂ = {b.J} AND F1₂ = {b.F1} AND F₂ = {int(b.F)} AND mF₂ = {int(b.mF)} AND I1₂ = {b.I1} AND I2₂ = {b.I2} AND P₂ = {P2} AND Ω₂ = {int(b.Omega)} AND state₂ = '{b.electronic_state}' "
    query += f"AND Px = {int(pol_vec[0])} AND Py = {int(pol_vec[1])} AND Pz = {int(pol_vec[2])} AND rme = {int(rme_only)}"
    return query

def check_states_in_ED_ME_coupled(Jg, Je, pol_vec):
    # load json
    path = Path(__file__).parent.parent / "pre_calculated"
    js = path / "precalculated.json"
    with open(js) as json_file:
        f = json.load(json_file)

    # check if ground state J is pre-cached   
    if not np.all(J in f['matrix_elements']['X'] for J in Jg):
        return False
    # check if excited state J is pre-cached
    if not np.all(J in f['matrix_elements']['B'] for J in Je):
        return False
    # check if the pol-vec is pre-cached
    if not list(pol_vec) in f['matrix_elements']['pol_vec']:
        return False
    else:
        return True
