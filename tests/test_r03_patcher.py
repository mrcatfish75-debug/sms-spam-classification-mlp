from pathlib import Path
import importlib.util
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('apply_r03', ROOT/'tools'/'apply_r03.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

EXAMPLE='''\
import gradio as gr

def analyze(text, mode, show_attention):
    return '<div>example</div>', '<span>detail</span>'

with gr.Blocks() as demo:
    sms_box = gr.Textbox(label="SMS")
    select_mode = gr.Radio(choices=['GRU', 'MLP', 'Compare'])
    use_attention = gr.Checkbox(label="Attention")
    run_button = gr.Button("Analyze")
    outcome = gr.HTML()
    details = gr.HTML()
    run_button.click(fn=analyze, inputs=[sms_box, select_mode, use_attention], outputs=[outcome, details])
demo.launch()
'''

class TestPatcher(unittest.TestCase):
    def test_detect(self):
        d=p.analyze(EXAMPLE)
        self.assertEqual(d['listeners'],['sms_box','select_mode','use_attention'])
        self.assertEqual(d['html_positions'],(0,1))
    def test_patch_parses(self):
        d=p.analyze(EXAMPLE)
        new=p.to_source_patch(EXAMPLE,d)
        compile(new,'app.py','exec')
        self.assertIn('sms_box.change(fn=_r03_clear_results',new)
        self.assertIn('use_attention.change(fn=_r03_clear_results',new)
        self.assertIn('guard_html(analyze, html_positions=(0, 1))',new)
    def test_fail_closed_when_unrecognized(self):
        with self.assertRaises(ValueError):p.analyze('print("nothing")')
    def test_refuse_repeat_patch(self):
        d=p.analyze(EXAMPLE)
        new=p.to_source_patch(EXAMPLE,d)
        with self.assertRaises(ValueError):p.to_source_patch(new,p.analyze(new))

if __name__=='__main__':unittest.main()
