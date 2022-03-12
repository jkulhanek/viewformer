import importlib
from aparse import click
from gettext import gettext as _


class LazyGroup(click.click.Group):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._commands = dict()

    def get_command(self, ctx, cmd_name):
        package = self._commands.get(cmd_name, None)
        if package is not None:
            if isinstance(package, str):
                package = importlib.import_module(package, __name__).main
            return package
        return None

    def list_commands(self, ctx):
        return list(self._commands.keys())

    def add_command(self, package_name, command_name=None):
        if command_name is not None:
            self._commands[command_name] = package_name
        else:
            self._commands[package_name.name] = package_name

    def format_commands(self, ctx, formatter) -> None:
        """Extra format methods for multi methods that adds all the commands
        after the options.
        """
        # allow for 3 times the default spacing
        commands = sorted(self._commands.keys())
        if len(commands):
            # limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
            rows = []
            for subcommand in commands:
                rows.append((subcommand, ''))

            with formatter.section(_("Commands")):
                formatter.write_dl(rows)
