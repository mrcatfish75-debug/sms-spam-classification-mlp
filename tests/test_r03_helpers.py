import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r03_helpers', ROOT/'payload'/'r03_helpers.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

class TestR03HTML(unittest.TestCase):
    def test_long_token_is_visually_bounded(self):
        v=helper.sanitize_generated_html('<div>'+('A'*2001)+'</div>')
        self.assertLess(len(v),160)
        self.assertIn('…',v)

    def test_token_escape(self):
        x=helper.safe_token_label('<img src=x onerror=alert(1)>')
        self.assertIn('&lt;img',x)
        self.assertNotIn('<img',x)

    def test_script_stripped(self):
        html='<div>normal<script>alert(1)</script><span>ok</span></div>'
        result=helper.sanitize_generated_html(html)
        self.assertNotIn('script',result)
        self.assertNotIn('alert(1)',result)
        self.assertIn('<span>ok</span>',result)

    def test_event_handler_stripped(self):
        result=helper.sanitize_generated_html('<div onmouseover="alert(1)">X</div>')
        self.assertNotIn('onmouseover',result)
        self.assertIn('>X</div>',result)

    def test_css_urls_stripped(self):
        result=helper.sanitize_generated_html('<div style="width: 50%; background-image:url(javascript:alert(1)); color: red">ok</div>')
        self.assertNotIn('background-image',result)
        self.assertIn('width: 50%',result)

    def test_preserve_aria_and_style(self):
        result=helper.sanitize_generated_html('<div class="bar" aria-label="Skor" style="width: 99%; background: #126bd0">OK</div>')
        self.assertIn('width: 99%',result)
        self.assertIn('aria-label',result)
        self.assertIn('background: #126bd0',result)

    def test_wrap_fn_preserves_signature(self):
        def predict(text, mode):
            return ('<div>'+text+'</div>','<span>ok</span>')
        wrapped=helper.guard_html(predict,(0,1))
        import inspect
        self.assertEqual(list(inspect.signature(wrapped).parameters),['text','mode'])
        a,b=wrapped('<img src=x onerror=alert(1)>','gru')
        self.assertNotIn('<img',a)
        self.assertEqual(b,'<span>ok</span>')

    def test_invalidation_message(self):
        self.assertIn('Hasil perlu diperbarui',helper.invalidated_html())

if __name__=='__main__':unittest.main()
