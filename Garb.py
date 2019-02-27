import graphviz
dot_data = tree.export_graphviz(lin_reg, out_file=None,
                     feature_names=X.columns[[0,1,2,3,4,6]],
                     class_names=X.columns,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree", format='png')
