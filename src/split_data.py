def load_lines(self):
    """
    Each line is a list of characters belonging to a specific language.
    Skipping the "/n" token.

    """

    with open(self.data_path, 'r') as f:
        lines = f.readlines()
    #lines = [list(paragraph)[:-1] for paragraph in lines]

    with open(self.label_path, 'r') as f:
        languages = f.readlines()
    languages = [language for language in languages]

    print('Loaded language paragraphs from: %s (%d)' % (self.data_path, len(lines)))

    lang_np = np.array(languages)
    print(lang_np.shape)
    #lines = np.array(lines)
    langs = list(set(languages))
    langs.sort()
    print(len(langs))
    val_set_idx = []
    train_set_idx = []
    for i, language in enumerate(langs):
        print(language)
        language_idx = np.where(lang_np==language)[0]
        print(language_idx.shape)
        random_set = np.random.permutation(500)
        val = language_idx[random_set[:100]]
        train = language_idx[random_set[100:]]
        val_set_idx.append(val)
        train_set_idx.append(train)
        print(i)


    #raise ValueError()
    with open("x_val_sub.txt", 'w') as f:
        for val_idx in val_set_idx:
            for idx in val_idx:
                para = lines[idx]
                f.write(para)

    with open("y_val_sub.txt", 'w') as f:
        for val_idx in val_set_idx:
            y_subset = lang_np[val_idx]
            for lang in y_subset:
                    f.write(lang)


    with open("x_train_sub.txt", 'w') as f:
        for train_idx in train_set_idx:
            for idx in train_idx:
                para = lines[idx]
                f.write(para)

    with open("y_train_sub.txt", 'w') as f:
        for train_idx in train_set_idx:
            y_subset = lang_np[train_idx]
            for lang in y_subset:
                    f.write(lang)

    return lines, languages
