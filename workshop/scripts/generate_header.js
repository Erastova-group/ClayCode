function  generateHeader () {
        fetch('../demonstrators.txt')
        .then(response => response.text())
        .then(data => {
            let demonstrators = data.split('\n').filter(line => line.trim() !== '');

            let oldDiv = document.getElementById('title');
            let title = oldDiv.textContent;
            // let newDiv = document.createElement('div');
            let div = document.createElement('div');
            div.className = 'box';
            //
            let h1 = document.createElement('h1');
            h1.id = 'title';
            h1.textContent = title;
            div.appendChild(h1);

            let pContact = document.createElement('p');
            pContact.innerHTML = '<strong>Contact:</strong><br/> Dr Valentina Erastova (<a href="mailto:valentina.erastova@ed.ac.uk">valentina.erastova@ed.ac.uk</a>)<br/> Office B21 (Office hours Wedn 10.00&dash;12.00 in weeks 3, 4 and 5)';
            div.appendChild(pContact);

            let pDemonstrators = document.createElement('p');
            pDemonstrators.innerHTML = '<strong>Demonstrators:</strong>';
            div.appendChild(pDemonstrators);

            let ul = document.createElement('ul');
            demonstrators.forEach(demonstrator => {
                ul.innerHTML += `<li>${demonstrator}</li>`;
            });
            div.appendChild(ul);

            oldDiv.parentNode.replaceChild(div, oldDiv);
        })
        .catch(error => console.error('Error:', error));

}
// window.onload = function () {
//         var title = document.getElementsByTagName('title')[0].innerText;
//         generateHeader(title);
// }
