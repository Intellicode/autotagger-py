from google.appengine.ext import webapp
from google.appengine.ext.webapp.util import run_wsgi_app
from autotagger.tagger import Tagger
import os
import cgi
from google.appengine.ext.webapp import template
class MainPage(webapp.RequestHandler):
    def get(self):


        template_values = {
          'tags': '',

          }

        path = os.path.join(os.path.dirname(__file__), 'index.html')
        self.response.out.write(template.render(path, template_values))
    def post(self):
        text =cgi.escape(self.request.get('content'))
        te = Tagger()
        tags = te.analyse_text(text,10)
        template_values = {
          'content':text,
          'tags': tags.toList(),
          'time': (te.getAlgorithmTime().microseconds/1000)
          }

        path = os.path.join(os.path.dirname(__file__), 'index.html')
        self.response.out.write(template.render(path, template_values))

application = webapp.WSGIApplication(
                                     [('/', MainPage)],
                                     debug=False)

def main():
  run_wsgi_app(application)

if __name__ == "__main__":
  main()