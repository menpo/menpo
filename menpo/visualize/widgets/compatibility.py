def add_class(widget, class_name):
    widget._dom_classes += (class_name,)


def remove_class(widget, class_name):
    new_class_list = list(widget._dom_classes)
    if class_name in new_class_list:
        new_class_list.remove(class_name)
    widget._dom_classes = tuple(new_class_list)
