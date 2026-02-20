import sys
sys.path.append('/home/lucas/projects/iWISDM')

from iwisdm.envs.shapenet.task_generator import TemporalTask
import iwisdm.envs.shapenet.task_generator as tg
import iwisdm.utils.helper as helper
from iwisdm import make

"""
DMS Tasks
"""
class DMSLoc(TemporalTask):
    """
    Compare objects on chosen frames are of the same location or not.
    @param: whens: a list of two frame names to compare stimuli location between
    """

    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(DMSLoc, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2 = self.whens[0], self.whens[1]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        # Get the locations of stimuli within each frame
        a1 = tg.GetLoc(objs1)
        a2 = tg.GetLoc(objs2)
        
        # Set operator to check if they're the same location
        if reverse:
            self._operator = tg.IsSame(a2, a1) 
        else:
            self._operator = tg.IsSame(a1, a2)

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

class DMSCategory(TemporalTask):
    """
    Compare objects on chosen frames are of the same category or not.
    @param: whens: a list of two frame names to compare stimuli category between
    """

    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(DMSCategory, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2 = self.whens[0], self.whens[1]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        # Get the category of stimuli within each frame
        a1 = tg.GetCategory(objs1)
        a2 = tg.GetCategory(objs2)
        
        # Set operator to check if they're the same category
        if reverse:
            self._operator = tg.IsSame(a2, a1) 
        else:
            self._operator = tg.IsSame(a1, a2)

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

class DMSObject(TemporalTask):
    """
    Compare objects on chosen frames are of the same object or not.
    @param: whens: a list of two frame names to compare stimuli object between
    """

    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(DMSObject, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2 = self.whens[0], self.whens[1]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        # Get the object of stimuli withsin each frame
        a1 = tg.GetObject(objs1)
        a2 = tg.GetObject(objs2)
        
        # Set operator to check if they're the same object
        if reverse:
            self._operator = tg.IsSame(a2, a1) 
        else:
            self._operator = tg.IsSame(a1, a2)

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

"""
DMS - New And
"""

class DMSA_LC(TemporalTask):
    """
    Compare objects on chosen frames are of the same location and category.
    @param: whens: a list of two frame names to compare stimuli category between
    """

    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(DMSA_LC, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2 = self.whens[0], self.whens[1]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        # Get the location of stimuli within each frame
        l1 = tg.GetLoc(objs1)
        l2 = tg.GetLoc(objs2)

        # Get the category of stimuli within each frame
        c1 = tg.GetCategory(objs1)
        c2 = tg.GetCategory(objs2)
        
        # Set operator to check if they're the same location and category
        if reverse:
            self._operator = tg.And(tg.IsSame(l2, l1), tg.IsSame(c2, c1))
        else:
            self._operator = tg.And(tg.IsSame(l1, l2), tg.IsSame(c1, c2))

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

class DMSA_LO(TemporalTask):
    """
    Compare objects on chosen frames are of the same location and object.
    @param: whens: a list of two frame names to compare stimuli category between
    """

    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(DMSA_LO, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2 = self.whens[0], self.whens[1]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        # Get the location of stimuli within each frame
        l1 = tg.GetLoc(objs1)
        l2 = tg.GetLoc(objs2)

        # Get the object of stimuli within each frame
        o1 = tg.GetObject(objs1)
        o2 = tg.GetObject(objs2)

        # Set operator to check if they're the same location and object
        if reverse:
            self._operator = tg.And(tg.IsSame(l2, l1), tg.IsSame(o2, o1))
        else:
            self._operator = tg.And(tg.IsSame(l1, l2), tg.IsSame(o1, o2))

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

"""
DMS - New OR
"""

class DMSO_LC(TemporalTask):
    """
    Compare objects on chosen frames are of the same location or category.
    @param: whens: a list of two frame names to compare stimuli category between
    """

    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(DMSO_LC, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2 = self.whens[0], self.whens[1]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        # Get the location of stimuli within each frame
        l1 = tg.GetLoc(objs1)
        l2 = tg.GetLoc(objs2)

        # Get the category of stimuli within each frame
        c1 = tg.GetCategory(objs1)
        c2 = tg.GetCategory(objs2)
        
        # Set operator to check if they're the same location or category
        if reverse:
            self._operator = tg.Or(tg.IsSame(l2, l1), tg.IsSame(c2, c1))
        else:
            self._operator = tg.Or(tg.IsSame(l1, l2), tg.IsSame(c1, c2))

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

class DMSO_LO(TemporalTask):
    """
    Compare objects on chosen frames are of the same location or object.
    @param: whens: a list of two frame names to compare stimuli category between
    """

    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(DMSO_LO, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2 = self.whens[0], self.whens[1]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        # Get the location of stimuli within each frame
        l1 = tg.GetLoc(objs1)
        l2 = tg.GetLoc(objs2)

        # Get the object of stimuli within each frame
        o1 = tg.GetObject(objs1)
        o2 = tg.GetObject(objs2)

        # Set operator to check if they're the same location or object
        if reverse:
            self._operator = tg.Or(tg.IsSame(l2, l1), tg.IsSame(o2, o1))
        else:
            self._operator = tg.Or(tg.IsSame(l1, l2), tg.IsSame(o1, o2))

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

"""
New CTX Tasks
"""

class ctxDM_OLC(TemporalTask):
    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(ctxDM_OLC, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2, when3 = self.whens[0], self.whens[1], self.whens[2]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        objs3 = tg.Select(when=when3)

        # Switch
        if reverse:
            condition = tg.IsSame(tg.GetObject(objs2), tg.GetObject(objs1))
            do_if_true = tg.IsSame(tg.GetLoc(objs3), tg.GetLoc(objs2))
            do_if_false = tg.IsSame(tg.GetCategory(objs3), tg.GetCategory(objs2))
        else:
            condition = tg.IsSame(tg.GetObject(objs1), tg.GetObject(objs2))
            do_if_true = tg.IsSame(tg.GetLoc(objs2), tg.GetLoc(objs3))
            do_if_false = tg.IsSame(tg.GetCategory(objs2), tg.GetCategory(objs3))

        switch = tg.Switch(condition, do_if_true, do_if_false, both_options_avail=False)
        self._operator = switch

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

class ctxDM_LOL(TemporalTask):
    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(ctxDM_LOL, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2, when3 = self.whens[0], self.whens[1], self.whens[2]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        objs3 = tg.Select(when=when3)

        # Switch
        if reverse:
            condition = tg.IsSame(tg.GetLoc(objs2), tg.GetLoc(objs1))
            do_if_true = tg.IsSame(tg.GetObject(objs3), tg.GetObject(objs2))
            do_if_false = tg.IsSame(tg.GetLoc(objs3), tg.GetLoc(objs2))
        else:
            condition = tg.IsSame(tg.GetLoc(objs1), tg.GetLoc(objs2))
            do_if_true = tg.IsSame(tg.GetObject(objs2), tg.GetObject(objs3))
            do_if_false = tg.IsSame(tg.GetLoc(objs2), tg.GetLoc(objs3))

        switch = tg.Switch(condition, do_if_true, do_if_false, both_options_avail=False)
        self._operator = switch

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

class ctxDM_COC(TemporalTask):
    def __init__(self, whens, first_shareable=None, reverse=False):
        # Initialize Class with parent class
        super(ctxDM_COC, self).__init__(whens=whens, first_shareable=first_shareable)

        # Get the whens
        when1, when2, when3 = self.whens[0], self.whens[1], self.whens[2]

        # Select the specified frames
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        objs3 = tg.Select(when=when3)

        # Switch
        if reverse:
            condition = tg.IsSame(tg.GetCategory(objs2), tg.GetCategory(objs1))
            do_if_true = tg.IsSame(tg.GetObject(objs3), tg.GetObject(objs2))
            do_if_false = tg.IsSame(tg.GetCategory(objs3), tg.GetCategory(objs2))
        else:
            condition = tg.IsSame(tg.GetCategory(objs1), tg.GetCategory(objs2))
            do_if_true = tg.IsSame(tg.GetObject(objs2), tg.GetObject(objs3))
            do_if_false = tg.IsSame(tg.GetCategory(objs2), tg.GetCategory(objs3))

        switch = tg.Switch(condition, do_if_true, do_if_false, both_options_avail=False)
        self._operator = switch

        # Set the number of frames
        self.n_frames = helper.compare_when([when1, when2]) + 1

"""
New 1-back And/Or Tasks
"""

def make_oneback_and_or(env, k, DMSA, reverse=False):
    """
    Create a one-back task with the specified parameters.
    @param: k: number of frames to include in the task
    @param: when1: first frame to compare
    @param: when2: second frame to compare
    @param: reverse: whether to reverse the comparison
    """

    # Code to make merged 1-back of length k
    tasks = []

    for _ in range(k):
        tasks.append(DMSA(whens=['last2', 'last0'], reverse=reverse))

    compo_infos = env.init_compositional_tasks(tasks)

    merged_info = compo_infos[0]
    for i in range(1, k):
        merged_info.frame_info.first_shareable = 2 + (i-1)*2

        info_to_merge = compo_infos[i]
        info_to_merge.frame_info.first_shareable = 0

        merged_info.merge(info_to_merge)

    return merged_info

"""
New 2-back tasks
"""
def make_twoback(env, k, DMS, reverse=False):
    """
    Create a two-back task with the specified parameters.
    @param: k: number of frames to include in the task
    @param: when1: first frame to compare
    @param: when2: second frame to compare
    @param: reverse: whether to reverse the comparison
    """

    # Code to make merged 2-back of length k
    tasks = []

    for _ in range(k):
        tasks.append(DMS(whens=['last4', 'last0'], reverse=reverse))

    compo_infos = env.init_compositional_tasks(tasks)

    merged_info = compo_infos[0]

    for i in range(1, k):
        merged_info.frame_info.first_shareable = 2 + (i-1)*2

        info_to_merge = compo_infos[i]
        info_to_merge.frame_info.first_shareable = 0

        merged_info.merge(info_to_merge)

    return merged_info

"""
New Interdms Tasks
"""

def make_interdms(env, task_type, DMS, reverse=False):
    if task_type == 'ABCABC':
        taskA = DMS(whens=['last6', 'last0'], first_shareable=2, reverse=reverse)
        taskB = DMS(whens=['last6', 'last0'], first_shareable=0, reverse=reverse)
        taskC = DMS(whens=['last6', 'last0'], first_shareable=0, reverse=reverse)

        taskA_info, taskB_info = env.init_compositional_tasks([taskA, taskB])
        taskA_info.merge(taskB_info)

        taskA_info.frame_info.first_shareable = 4

        taskC_info, = env.init_compositional_tasks([taskC])
        taskA_info.merge(taskC_info)
    elif task_type == 'ABBCCA':
        taskA = DMS(whens=['last10', 'last0'], first_shareable=2, reverse=reverse)
        taskB = DMS(whens=['last2', 'last0'], first_shareable=0, reverse=reverse)
        taskC = DMS(whens=['last2', 'last0'], first_shareable=0, reverse=reverse)

        taskA_info, taskB_info = env.init_compositional_tasks([taskA, taskB])
        taskA_info.merge(taskB_info)

        taskA_info.frame_info.first_shareable = 6

        taskC_info, = env.init_compositional_tasks([taskC])
        taskA_info.merge(taskC_info)

    return taskA_info
