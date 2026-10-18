"""检查电网与电池状态可独立组合，并与看板字段一致。"""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from main import Optimizer, config


class OperationStateTests(unittest.TestCase):
    def test_independent_states_are_written(self):
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.db = Mock()
        for grid_mode, (buy, sell) in enumerate(((100, 0), (0, 100), (10, 10)), 1):
            for battery_mode, (charge, discharge) in enumerate(
                ((100, 0), (0, 100), (10, 10)), 1
            ):
                with self.subTest(grid=grid_mode, battery=battery_mode):
                    optimizer.db.reset_mock()
                    result = {'p_b': buy, 'p_s': sell, 'p_ch': charge,
                              'p_dh': discharge, 'p_ab': 0}
                    optimizer._write_optimization_results(
                        0, result, 0.5, 100, 100, 100, 100,
                    )
                    calls = optimizer.db.write_point.call_args_list
                    fields = calls[0].args[2]
                    self.assertEqual(fields['grid_mode'], grid_mode)
                    self.assertEqual(fields['battery_mode'], battery_mode)
                    self.assertEqual(fields['p_dh'], -discharge)
                    self.assertEqual(fields['p_sell'], -sell)
                    self.assertNotIn('mode', fields)
                    self.assertEqual(calls[2].args[1], config.MEAS_MODE)
                    self.assertEqual(calls[2].args[2], {
                        'grid_mode': grid_mode, 'battery_mode': battery_mode,
                        'time_slot': 0,
                    })

    def test_dashboard_uses_separate_states(self):
        path = Path(__file__).resolve().parents[1] / 'ems_dashboard.json'
        dashboard = json.loads(path.read_text(encoding='utf-8'))['spec']
        for panel_id, field, labels in (
            ('panel-1', 'grid_mode', ['购电', '售电', '无交换']),
            ('panel-9', 'battery_mode', ['充电', '放电', '待机']),
        ):
            panel = dashboard['elements'][panel_id]['spec']
            query = panel['data']['spec']['queries'][0]['spec']['query']['spec']
            self.assertTrue(query['rawQuery'])
            self.assertIn(f'last("{field}")', query['query'])
            mappings = panel['vizConfig']['spec']['fieldConfig']['defaults']['mappings']
            self.assertEqual(
                [mappings[0]['options'][str(i)]['text'] for i in range(1, 4)], labels,
            )
        items = dashboard['layout']['spec']['items']
        self.assertEqual(sum(
            item['spec']['element']['name'] == 'panel-9' for item in items
        ), 1)


if __name__ == '__main__':
    unittest.main()
