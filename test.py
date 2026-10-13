import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

m = pyo.ConcreteModel()
m.x = pyo.Var(domain=pyo.NonNegativeReals)
m.y = pyo.Var(domain=pyo.NonNegativeReals)

m.balance = pyo.Constraint(expr=m.x + m.y >= 10)
m.cost = pyo.Objective(
    expr=2 * m.x + 3 * m.y,
    sense=pyo.minimize,
)

solver = pyo.SolverFactory("gurobi_direct")
result = solver.solve(m, tee=True)
if result.solver.termination_condition == TerminationCondition.optimal:
    print("x =", pyo.value(m.x))
    print("y =", pyo.value(m.y))
    print("最小成本 =", pyo.value(m.cost))
else:
    print("求解状态：", result.solver.termination_condition)
    