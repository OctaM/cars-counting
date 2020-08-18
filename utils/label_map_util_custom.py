
def load_labelmap(path):
    categories = []
    with open(path) as label_file:
        for line in label_file:
            items = line.split()
            category = ''
            index, category = int(items[0]), category.join(items[1:])
            category = category.rstrip()
            categories.append({"id": index, 'name': category})
    return categories


def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index