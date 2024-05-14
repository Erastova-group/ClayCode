function checkURL(URL, name) {
    var request = new XMLHttpRequest();
    request.open('GET', URL, true);
    request.onreadystatechange = function(){
        if (request.readyState === 4){
            if (request.status === 200) {
                // The page exists, you can create the link
                var aTag = document.createElement('a');
                aTag.setAttribute('href', URL);
                aTag.innerText = name;
                document.body.appendChild(aTag);
            } else {
                    var textNode = document.createTextNode(name);
                    document.body.appendChild(textNode);
            }
        }
    };
    request.send();
}
