

# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

class BaseObjState:
    def __init__(self, obj, robot):
        self.obj = obj
        self.robot = robot
        self.obj_name = obj.name
        self.size = obj.size

    @staticmethod
    def get_dependencies():
        return []

    def get_value(self):
        pass
