'''
Class lifted from https://github.com/nowacklab/Nowack_Lab/blob/master/Utilities/save.py
'''
from jsonpickle.ext import numpy as jspnp
import json, os, pickle, bz2, jsonpickle as jsp, numpy as np, re, sys, subprocess
from datetime import datetime
jspnp.register_handlers()
from copy import copy
import h5py, glob, matplotlib, inspect, platform, hashlib, shutil, time
import matplotlib.pyplot as plt, socket
from scipy.sparse import spmatrix

'''
How saving and loading works:
1) Walks through object's __dict__, subdictionaries, and subobjects, picks out
numpy arrays, and saves them in a hirearchy in HDF5. Dictionaries are
represented as groups in HDF5, and objects are also represented as groups, but
with a ! preceding the name. This is parsed when loading.
2) All numpy arrays and matplotlib objects in the dictionary hierarchy are set
to None, and the object is saved to JSON.
3) The saved object is immediately reloaded to see if everything went well.
3a) First, the JSON file is loaded to set up the dictionary hierarchy.
3b) Second, we walk through the HDF5 file (identifying objects and dictionaries
as necessary) and populate the numpy arrays.
'''


class Simulation:
    _daq_inputs = [] # DAQ input labels expected by this class
    _daq_outputs = [] # DAQ input labels expected by this class
    instrument_list = []
    fig = None
    interrupt = False # boolean variable used to interrupt loops in the do.

    def __init__(self, instruments = {}):
        self.make_timestamp_and_filename()
        self._load_instruments(instruments)

    def __getstate__(self):
        '''
        Returns a dictionary of everything we want to save to JSON.
        This excludes numpy arrays which are saved to HDF5
        '''
        def walk(d):
            d = d.copy() # make sure we don't modify original dictionary
            variables = list(d.keys()) # list of all the variables in the dictionary

            for var in variables: # copy so we don't change size of array during iteration
                ## Don't save numpy arrays to JSON
                if type(d[var]) is np.ndarray:
                    d[var] = None
                elif isinstance(d[var], spmatrix):
                    d[var] = None

                ## Don't save matplotlib objects to JSON
                if hasattr(d[var], '__module__'): # built-in types won't have __module__
                    m = d[var].__module__
                    m = m[:m.find('.')] # will strip out "matplotlib", if there
                    if m == 'matplotlib':
                        d[var] = None
                elif type(d[var]) is list: # check for lists of mpl objects
                    if hasattr(d[var][0], '__module__'): # built-in types won't have __module__
                        m = d[var][0].__module__
                        m = m[:m.find('.')] # will strip out "matplotlib"
                        if m == 'matplotlib':
                            d[var] = None




                ## Walk through dictionaries
                if isinstance(d[var], dict):
                    d[var] = walk(d[var]) # This unfortunately erases the dictionary...

            return d # only return ones that are the right type.

        return walk(self.__dict__)


    def __setstate__(self, state):
        '''
        Default method for loading from JSON.
        `state` is a dictionary.
        '''
        self.__dict__.update(state)

    def _load_hdf5(self, filename, unwanted_keys = []):
        '''
        Loads data from HDF5 files. Will walk through the HDF5 file and populate
        the object's dictionary and subdictionaries (already loaded by JSON)
        '''
        with h5py.File(filename, 'r') as f:
            def walk(d, f):
                for key in f.keys():
                    if key not in unwanted_keys:
                        # check if it's a dictionary or object
                        if f.get(key, getclass=True) is h5py._hl.group.Group:
                            if key[0] == '!': # it's an object
                                walk(d[key[1:]].__dict__, f[key])
                                # [:1] strips the !; walks through the subobject
                            else: # it's a dictionary
                                # walk through the subdictionary
                                walk(d[key], f[key])
                        else:
                            d[key] = f[key][:] # we've arrived at a dataset

                    ## If a dictionary key was an int, convert it back
                    try:
                        newkey = int(key) # test if can convert to an integer
                        value = d.pop(key) # replace the key with integer version
                        d[newkey] = value # do this all stepwise in case of error
                    except:
                        pass

                return d

            walk(self.__dict__, f) # start walkin'


    def _load_instruments(self, instruments={}):
        '''
        Loads instruments from a dictionary.
        '''
        for instrument in instruments:
            setattr(self, instrument, instruments[instrument])
            if instrument == 'daq':
                for ch in self._daq_inputs:
                    if ch not in self.daq.inputs:
                        raise Exception('Need to set daq input labels! Need a %s' %ch)
                for ch in self._daq_outputs:
                    if ch not in self.daq.outputs:
                        raise Exception('Need to set daq output labels! Need a %s' %ch)


    @staticmethod
    def _load_json(json_file, unwanted_keys = []):
        '''
        Loads an object from JSON.
        '''
        with open(json_file, encoding='utf-8') as f:
            obj_dict = json.load(f)

        def walk(d):
            '''
            Walk through dictionary to prune out Instruments
            '''
            for key in list(d.keys()): # convert to list because dictionary changes size
                if key in unwanted_keys: # get rid of keys you don't want to load
                    d[key] = None
                elif 'py/' in key:
                    if 'py/object' in d:
                        if 'Instruments' in d['py/object']: # if this is an instrument
                            d = None # Don't load it.
                            break
                    elif 'py/id' in d: # This is probably another instance of an Instrument.
                        d = None # Don't load it.
                        break
                if isinstance(d[key], dict):
                    d[key] = walk(d[key])
            return d

        obj_dict = walk(obj_dict)

        # If the class of the object is custom defined in __main__ or in a
        # different branch, then just load it as a Simulation.
        try:
            exec(obj_dict['py/object']) # see if class is in the namespace
        except:
            obj_dict['py/object'] = 'viscous_flow_graphene.navier_stokes_grid.ViscousSimulation'


        # Decode with jsonpickle.
        obj_string = json.dumps(obj_dict)
        obj = jsp.decode(obj_string)

        return obj

    def _save(self, filename=None, savefig=True, extras=False, ignored = []):
        '''
        Saves data. numpy arrays are saved to one file as hdf5, everything else
        is saved to JSON
        Keyword arguments:
        filename -- The path where the datafile will be saved. One hdf5 file and
        one JSON file with the specified filename will be saved. If no filename
        is supplied then the filename is generated from the timestamp
        savefig -- If "True" figures are saved. If "False" only data and config
        are saved
        extras -- If True, saved to a subfolder of the current data directory
        named "extras" to reduce clutter. If False, saved in the normal way.
        ignored -- Array of objects to be ignored during saving. Passed to
        _save_hdf5 and _save_json.
        Saved data stored locally but also copied to the data server.
        If you specify no filename, one will be automatically generated.
        Locally, saves to ~/data/; remotely, saves to /labshare/data/
        If you specify a relative (partial) path, the data will still be saved
        in the data directory, but in subdirectories specified in the filename.
        For example, if the filename is 'custom/custom_filename', the data will
        be saved in (today's data directory)/custom/custom_filename.
        If you specify a full path, the object will save to that path locally
        and to [get_data_server_path()]/[get_computer_name()]/other/[hirearchy past root directory]
        '''

        # Saving to the experiment-specified directory
        if filename is None:
            if not hasattr(self, 'filename'):  # if you did not make a filename
                self.make_timestamp_and_filename()
            filename = self.filename

        if extras is True:
            extras = 'extras'
        else:
            extras = ''

        if os.path.dirname(filename) == '':  # you specified a filename with no preceding path
            local_path = os.path.join(get_local_data_path(),
                                      get_todays_data_dir(),
                                      extras,
                                      filename)
            remote_path = os.path.join(get_remote_data_path(),
                                       get_todays_data_dir(),
                                       extras,
                                       filename)

        # Saving to a custom path
        else: # you specified some sort of path
            local_path = filename
            remote_path = os.path.join(get_remote_data_path(), '..', 'other', *filename.replace('\\', '/').split('/')[1:]) # removes anything before the first slash. e.g. ~/data/stuff -> data/stuff
            # All in all, remote_path should look something like: .../labshare/data/bluefors/other/

        # Save locally:
        local_dir = os.path.split(local_path)[0]
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        self._save_hdf5(local_path, ignored = ignored)
        self._save_json(local_path)

        nopdf = True
        if savefig and self.fig is not None:
            self.fig.savefig(local_path+'.pdf', bbox_inches='tight')
            nopdf = False

        ## See if saving worked properly
        try:
            self.load(local_path)
        except:
            raise Exception('Reloading failed, but object was saved!')


    def _save_hdf5(self, filename, ignored = []):
        '''
        Save numpy arrays to h5py. Walks through the object's dictionary
        and any subdictionaries and subobjects, picks out numpy arrays,
        and saves them in the hierarchical HDF5 format.
        A subobject is designated by a ! at the beginning of the varaible name.
        '''

        with h5py.File(filename+'.h5', 'w') as f:
            # Walk through the dictionary
            def walk(d, group):
                for key, value in d.items():
                    # If the key is in ignored then skip over it
                    if key in ignored:
                        continue
                    if type(key) is int:
                        key = str(key) # convert int keys to string. Will be converted back when loading
                    # If a numpy array is found
                    if type(value) is np.ndarray:
                        # Save the numpy array as a dataset
                        d = group.create_dataset(key, value.shape,
                            compression = 'gzip', compression_opts=9)
                        d.set_fill_value = np.nan
                        # Fill the dataset with the corresponding value
                        d[...] = value
                    elif isinstance(value, spmatrix):
                        continue
                    # If a dictionary is found
                    elif isinstance(value, dict):
                        new_group = group.create_group(key) # make a group with the dictionary name
                        walk(value, new_group) # walk through the dictionary
                    # If the there is some other object
                    elif hasattr(value, '__dict__'):
                        if isinstance(value, Simulation): # restrict saving Simulations.
                            new_group = group.create_group('!'+key) # make a group with !(object name)
                            walk(value.__dict__, new_group) # walk through the object dictionary

            walk(self.__dict__, f)


    def _save_json(self, filename):
        '''
        Saves an object to JSON. Specify a custom filename,
        or use the `filename` variable under that object.
        Through __getstate__, ignores any numpy arrays when saving.
        '''
        if not exists(filename+'.json'):
            obj_string = jsp.encode(self)
            obj_dict = json.loads(obj_string)
            with open(filename+'.json', 'w', encoding='utf-8') as f:
                json.dump(obj_dict, f, sort_keys=True, indent=4)

    def do(self):
        '''
        Do the main part of the Simulation. Write this function for subclasses.
        run() wraps this function to enable keyboard interrupts.
        run() also includes saving and elapsed time logging.
        '''
        print('Doing stuff!')


    @classmethod
    def load(cls, filename=None, instruments={}, unwanted_keys=[]):
        '''
        Basic load method. Calls _load_json, not loading instruments, then loads from HDF5, then loads instruments.
        Overwrite this for each subclass if necessary.
        Pass in an array of the names of things you don't want to load.
        By default, we won't load any instruments, but you can pass in an instruments dictionary to load them.
        Filename may be None (load last Simulation), a filename, a path, or an
        index (will search all Simulations of this type for this experiment)
        '''

        if filename is None: # tries to find the last saved object; not guaranteed to work
            filename = -1
        if type(filename) is int:
            folders = list(glob.iglob(os.path.join(get_local_data_path(), get_todays_data_dir(),'..','*')))
            # Collect a list of all Simulations of this type for this experiment
            l = []
            for i in range(len(folders)):
                l = l + list(glob.iglob(os.path.join(folders[i],'*_%s.json' %cls.__name__)))
            try:
                filename = l[filename] # filename was an int
            except:
                pass
            if type(filename) is int:  # still
                raise Exception('could not find %s to load.' %cls.__name__)
        elif os.path.dirname(filename) == '': # if no path specified
            os.path.join(get_local_data_path(), get_todays_data_dir(), filename)

        # Remove file extensions
        filename = os.path.splitext(filename)[0]

        obj = Simulation._load_json(filename+'.json', unwanted_keys)
        obj._load_hdf5(filename+'.h5')
        obj._load_instruments(instruments)
        return obj

    def make_timestamp_and_filename(self):
        '''
        Makes a timestamp and filename from the current time.
        '''
        now = datetime.now()
        self.timestamp = now.strftime("%Y-%m-%d %I:%M:%S %p")
        self.filename = now.strftime('%Y-%m-%d_%H%M%S')
        self.filename += '_' + self.__class__.__name__


    def plot(self):
        '''
        Update all plots.
        '''
        if self.fig is None:
            self.setup_plots()


    def run(self, plot=True, **kwargs):
        '''
        Wrapper function for do() that catches keyboard interrrupts
        without leaving open DAQ tasks running. Allows scans to be
        interrupted without restarting the python instance afterwards
        Keyword arguments:
            plot: boolean; to plot or not to plot?
        Check the do() function for additional available kwargs.
        '''
        self.interrupt = False
        done = None

        ## Before the do.
        if plot:
            self.setup_plots()
        time_start = time.time()

        ## The do.
        try:
            done = self.do(**kwargs)
        except KeyboardInterrupt:
            self.interrupt = True

        ## After the do.
        time_end = time.time()
        self.time_elapsed_s = time_end-time_start

        if self.time_elapsed_s < 60: # less than a minute
            t = self.time_elapsed_s
            t_unit = 'seconds'
        elif self.time_elapsed_s < 3600: # less than an hour
            t = self.time_elapsed_s/60
            t_unit = 'minutes'
        else:
            t = self.time_elapsed_s/3600
            t_unit = 'hours'
        # Print elapsed time e.g. "Scanplane took 2.3 hours."
        print('%s took %.1f %s.' %(self.__class__.__name__, t, t_unit))
        self.save()

        # If this run is in a loop, then we want to raise the KeyboardInterrupt
        # to terminate the loop.
        if self.interrupt:
            raise KeyboardInterrupt

        return done

    def save(self, filename=None, savefig=True, extras=False):
        '''
        Basic save method. Just calls _save. Overwrite this for each subclass.
        '''
        self._save(filename, savefig=savefig, extras=extras)


    def setup_plots(self):
        '''
        Set up all plots.
        '''
        self.fig, self.ax = plt.subplots() # example: just one figure



def exists(filename):
    inp='y'
    if os.path.exists(filename+'.json'):
        inp = input('File %s already exists! Overwrite? (y/n)' %(filename+'.json'))
    if inp not in ('y','Y'):
        print('File not saved!')
        return True
    return False

def get_computer_name():
    computer_name = socket.gethostname()
    aliases = {'SPRUCE': 'bluefors', 'HEMLOCK': 'montana'} # different names we want to give the directories for each computer
    if computer_name in aliases.keys():
        computer_name = aliases[computer_name]
    return computer_name


def get_experiment_data_dir():
    '''
    Finds the most recently modified (current) experiment data directory. (Not the full path)
    '''

    latest_subdir = max(glob.glob(os.path.join(get_local_data_path(), '*/')), key=os.path.getmtime)

    return os.path.relpath(latest_subdir, get_local_data_path()) # strip just the directory name

    ## If we're sure that there will only be one directory per date. Bad assumption.
    # exp_dirs = []
    # for name in os.listdir(get_local_data_path()): # all experiment directories
    #     if re.match(r'\d{4}-', name[:5]): # make sure directory starts with "20XX-"
    #         exp_dirs.append(name)

    # exp_dirs.sort(key=lambda x: datetime.strptime(x[:10], '%Y-%m-%d')) # sort by date
    #
    # return exp_dirs[-1] # this is the most recent


def get_data_server_path():
    '''
    Returns full path of the data server's main directory, formatted based on OS.
    '''
    if platform.system() == 'Windows':
        return r'\\SAMBASHARE\labshare\data'
    elif platform.system() == 'Darwin': # Mac
        return '/Volumes/labshare/data/'
    elif platform.system() == 'Linux':
        return '/mnt/labshare/data/'
    else:
        raise Exception('What OS are you using?!? O_O')


def get_local_data_path():
    '''
    Returns full path of the local data directory.
    '''
    return os.path.join(
                os.path.expanduser('~'),
                'data',
                get_computer_name(),
                'experiments'
            )


def get_remote_data_path():
    '''
    Returns full path of the remote data directory.
    '''
    return os.path.join(
                get_data_server_path(),
                get_computer_name(),
                'experiments'
            )


def get_todays_data_dir():
    '''
    Returns name of today's data directory.
    '''
    experiment_path = get_experiment_data_dir()
    now = datetime.now()
    todays_data_path = os.path.join(experiment_path, now.strftime('%Y-%m-%d'))

    # Make local and remote directory
    dirs = [get_local_data_path()]
    if os.path.exists(get_data_server_path()):
        dirs += [get_remote_data_path()]
    for d in dirs:
        try:
            filename = os.path.join(d, todays_data_path)
            if not os.path.exists(filename):
                os.makedirs(filename)
        except Exception as e:
            print(e)

    return todays_data_path

def open_experiment_data_dir():
    filename = get_local_data_path()
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

def set_experiment_data_dir(description=''):
    '''
    Run this when you start a new experiment (e.g. a cooldown).
    Makes a new directory in the data folder corresponding to your computer
    with the current date and a description of the experiment.
    '''
    now = datetime.now()
    now_fmt = now.strftime('%Y-%m-%d')

    # Make local and remote directories:
    dirs = [get_local_data_path()]
    if os.path.exists(get_data_server_path()):
        dirs += [get_remote_data_path()]
    for d in dirs:
        try:
            filename = os.path.join(d, now_fmt + '_' + description)
            if not os.path.exists(filename):
                os.makedirs(filename)
        except:
            print('Error making directory %s' %d)


def _md5(filename):
    '''
    Calculates an MD5 checksum for the given file
    '''
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
