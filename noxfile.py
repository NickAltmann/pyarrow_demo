import nox


@nox.session
def tests(session):
    session.install("pip", "pytest")
    session.run("pip", "install", ".", "-v")
    #session.run("pytest")
    session.run("python", "run.py")
