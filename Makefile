gh-pages:
	tar czf ../html.tgz -C docs/build/html .
	git checkout gh-pages
	rm -rf *
	tar xzf ../html.tgz
	git add .
	git commit -a -m "publish the docs"
	git push origin gh-pages
	git checkout master
