function adjustStyle() {
    console.log('adjustStyle function triggered');
    let linkElements = document.getElementsByTagName('link');
    let button = document.getElementById('dyslexiaButton');
    for (let i = 0; i < linkElements.length; i++) {
        console.log(linkElements[i].getAttribute('href'));
        if (linkElements[i].getAttribute('href') !== null && linkElements[i].getAttribute('href').endsWith('scripts/style.css')) {
            console.log('change to dyslexia style');
            let linkStart = linkElements[i].getAttribute('href').match(/.*(?=style.css)/);
            linkElements[i].setAttribute('href', linkStart + 'style_dyslexia.css');
            button.innerHTML = 'Regular display';
            localStorage.setItem('dyslexiaMode', 'true');
        } else if (linkElements[i].getAttribute('href') !== null && linkElements[i].getAttribute('href').endsWith('scripts/style_dyslexia.css')) {
            console.log('change to regular style');
            let linkStart = linkElements[i].getAttribute('href').match(/.*(?=style_dyslexia.css)/);

            linkElements[i].setAttribute('href', linkStart + 'style.css');
            button.innerHTML = 'Dyslexia-friendly display';
            localStorage.setItem('dyslexiaMode', 'false');
        }
    }
    switchPrintStyle()
}

function switchPrintStyle() {
    console.log('switchToDyslexic function triggered');
    let linkElements = document.getElementsByTagName('link');
    for (let i = 0; i < linkElements.length; i++) {
        console.log(linkElements[i].getAttribute('href'));
        if (localStorage.getItem('dyslexiaMode') === 'true' && linkElements[i].getAttribute('href') !== null && linkElements[i].media === 'print' && linkElements[i].getAttribute('href').endsWith('style.css')) {
            console.log('change to dyslexia style');
            let linkStart = linkElements[i].getAttribute('href').match(/.*(?=style.css)/);
            linkElements[i].setAttribute('href', linkStart + 'style_dyslexia.css');
            localStorage.setItem('dyslexiaMode', 'true');
        } else if (localStorage.getItem('dyslexiaMode') === 'false' && linkElements[i].getAttribute('href') !== null && linkElements[i].media === 'print' && linkElements[i].getAttribute('href').endsWith('style_dyslexia.css')) {
            console.log('change to regular style');
            let linkStart = linkElements[i].getAttribute('href').match(/.*(?=style_dyslexia.css)/);
            linkElements[i].setAttribute('href', linkStart + 'style.css');
            localStorage.setItem('dyslexiaMode', 'false');
        }
    }
}

function adjustForDyslexia() {
    let button = document.createElement('button');
    button.className = 'dyslexiaButton';
    button.id = 'dyslexiaButton';
    button.innerHTML = 'Dyslexia-friendly display';
    document.body.appendChild(button);
    if (localStorage.getItem('dyslexiaMode') !== 'null') {
        localStorage.setItem('dyslexiaMode', 'false');
    }
    button.addEventListener('click', function () {
        adjustStyle();
    });
    window.addEventListener('beforeprint', function () {
        button.style.display = 'none';
    });
    window.addEventListener('afterprint', function () {
        button.style.display = 'block';
    });
}

// function adjustForDyslexia() {
//     let button = document.createElement('button');
//     button.className = 'dyslexiaButton';
//     button.id = 'dyslexiaButton';
//     button.innerHTML = 'Dyslexia-friendly display';
//     document.body.appendChild(button);
//     button.addEventListener('click', function () {
//         adjustStyle();
//     });
//     // let observer = new MutationObserver(function (mutations) {
//     //     mutations.forEach(function (mutation) {
//     //         if (mutation.type === 'childList') {
//     //             adjustStyle();
//     //         }
//     //     });
//     // });
//     // observer.observe(document.head, {childList: true});
//     window.addEventListener('beforeprint', function () {
//         button.style.display = 'none';
//         adjustStyle();
//     });
// }



