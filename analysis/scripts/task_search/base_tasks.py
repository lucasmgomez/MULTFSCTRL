import sys
sys.path.append('/home/lucas/projects/iWISDM')

from iwisdm.envs.shapenet.task_generator import TemporalTask
import iwisdm.envs.shapenet.registration as env_reg
import iwisdm.envs.shapenet.task_generator as tg
import iwisdm.utils.helper as helper

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

# DMSObject(whens=['last2', 'last0'], reverse=reverse)

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
WRITE TASKS TO JSON
"""
from iwisdm import make
from iwisdm import read_write
import json
import os
import shutil

stim_dir = '/mnt/tempdata/lucas/multi_task_data/stim/shapenet_fmri_fixed' 

# Create environment
env = make(
    env_id='ShapeNet',
    dataset_fp=stim_dir,
)

# Initialize environment
print(env.env_spec.MAX_DELAY)

# Create task instances
reverse = False
dms_loc = DMSLoc(whens=['last2', 'last0'], reverse=reverse)
dms_ctg = DMSCategory(whens=['last2', 'last0'], reverse=reverse)
dms_obj = DMSObject(whens=['last2', 'last0'], reverse=reverse)
ctxdm_olc = ctxDM_OLC(whens=['last4', 'last2', 'last0'], reverse=reverse)
ctxdm_lol = ctxDM_LOL(whens=['last4', 'last2', 'last0'], reverse=reverse)
ctxdm_coc = ctxDM_COC(whens=['last4', 'last2', 'last0'], reverse=reverse)

# Save tasks to JSON
read_write.write_task(dms_loc , '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/base_tasks/dms_loc.json')
read_write.write_task(dms_ctg , '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/base_tasks/dms_ctg.json')
read_write.write_task(dms_obj , '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/base_tasks/dms_obj.json')
read_write.write_task(ctxdm_olc , '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/base_tasks/ctxdm_olc.json')
read_write.write_task(ctxdm_lol , '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/base_tasks/ctxdm_lol.json')
read_write.write_task(ctxdm_coc , '/home/lucas/projects/MULTFSCTRL/analysis/scripts/task_search/base_tasks/ctxdm_coc.json') 