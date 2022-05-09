import nox

@nox.session
def format(session):
    session.install('black')
    session.run('black', 'src/eval_sel/')

@nox.session
def static_typing(session):
    session.install('poetry')
    session.run('poetry', 'install')
    session.run('poetry', 'run', 'mypy', 'src/eval_sel/')

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'src/eval_sel/')

@nox.session
def tests(session):
    session.install('poetry')
    session.run('poetry', 'install')
    session.run('poetry', 'run', 'pytest')