# Starter code prepared by James Hays for CS 143, Brown University

# This function creates a webpage (html and images) visualizing the
# classiffication results. This webpage will contain
# (1) A confusion matrix plot
# (2) A table with one row per category, with 3 columns - training
# examples, true positives, false positives, and false negatives.

# false positives are instances claimed as that category but belonging to
# another category, e.g. in the 'forest' row an image that was classified
# as 'forest' but is actually 'mountain'. This same image would be
# considered a false negative in the 'mountain' row, because it should have
# been claimed by the 'mountain' classifier but was not.

# This webpage is similar to the one we created for the SUN database in
# 2010: http://people.csail.mit.edu/jxiao/SUN/classification397.html

def create_results_webpage(train_image_paths, test_image_paths,
                           train_labels, test_labels, categories, abbr_categories, predicted_categories):
    pass


