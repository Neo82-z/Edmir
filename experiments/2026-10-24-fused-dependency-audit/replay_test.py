import unittest
from pathlib import Path
from replay import replay,read_capture

def launch(source,stream):
    return f'fused_cutlass device=0 0 1 window {source} 1 1 1 b c workspace {stream}'

class EventSemantics(unittest.TestCase):
    def test_event_captures_entire_prior_stream(self):
        order=[(0,0,0),(1,0,0),(2,0,0)]
        a,_=replay([launch(0,'a'),launch(0,'a'),'record_event device=0 e a',
            'wait_event device=0 b e 0',launch(0,'b'),
            'sync_stream device=0 a','sync_stream device=0 b'],order)
        self.assertEqual(a[2],{0,1})

    def test_event_rerecord_does_not_retarget_previous_wait(self):
        order=[(0,0,0),(1,0,0),(2,0,0)]
        a,_=replay([launch(0,'a'),'record_event device=0 e a',
            'wait_event device=0 b e 0',launch(0,'a'),
            'record_event device=0 e a',launch(0,'b'),
            'sync_stream device=0 a','sync_stream device=0 b'],order)
        self.assertEqual(a[2],{0});self.assertNotIn(1,a[2])

    def test_wait_applies_to_all_future_work(self):
        order=[(0,0,0),(1,0,0),(2,0,0)]
        a,_=replay([launch(0,'a'),'record_event device=0 e a',
            'wait_event device=0 b e 0',launch(0,'b'),launch(0,'b'),
            'sync_stream device=0 a','sync_stream device=0 b'],order)
        self.assertEqual(a[1],{0});self.assertEqual(a[2],{0,1})

    def test_host_sync_orders_subsequent_launch(self):
        a,_=replay([launch(0,'a'),'sync_stream device=0 a',launch(0,'b'),
            'sync_stream device=0 b'],[(0,0,0),(1,0,0)])
        self.assertEqual(a[1],{0})

    def test_disjoint_streams_are_not_forced_into_host_submission_order(self):
        a,_=replay([launch(0,'a'),launch(0,'b'),'sync_stream device=0 a',
            'sync_stream device=0 b'],[(0,0,0),(1,0,0)])
        self.assertEqual(a[1],set())

    def test_actual_event_removal_changes_graph(self):
        order,sections=read_capture(Path(__file__).parent/'prepared-identity-full.txt')
        a,_=replay(sections['ITER 0'],order)
        b,_=replay([s for s in sections['ITER 0'] if not s.startswith('wait_event ')],order)
        self.assertNotEqual(a,b)
        self.assertEqual(sum(map(len,b.values())),24) # 4 streams × C(4,2)

if __name__=='__main__':unittest.main()
