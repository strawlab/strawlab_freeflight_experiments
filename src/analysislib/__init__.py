import progressbar

def get_progress_bar(name, maxval):
    widgets = ["%s: " % name, progressbar.Percentage(),
               progressbar.Bar(), progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,maxval=maxval).start()
    return pbar
