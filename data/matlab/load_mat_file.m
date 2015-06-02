function obj = load_mat_file(path)
    inf = whos('-file',path);
    dat = load(path, '-mat');
    obj = dat.(inf.name);
end
