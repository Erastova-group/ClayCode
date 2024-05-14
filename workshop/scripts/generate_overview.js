function generateOverview  () {
    var toc = document.createElement('p');
    toc.id = 'toc';
    toc.style.fontSize = '20px';
    toc.style.fontWeight = 'normal';
    toc.style.marginLeft = '20px';
    toc.style.marginTop = '0';
    toc.style.lineHeight = '1.5';

    var h2Elements = document.getElementsByTagName('h2');
    for (var i = 0; i < h2Elements.length; i++) {
        if (h2Elements[i].id.startsWith('toc_')) {
            var tocItem = document.createElement('a');
            tocItem.href = '#' + h2Elements[i].id;
            var idNumber = h2Elements[i].id.match(/\d+$/)[0];
            tocItem.innerText = idNumber + '. ' + h2Elements[i].innerText;
            toc.appendChild(tocItem);
            toc.appendChild(document.createElement('br'));
        }
    }


    var tasks = document.createElement('h3');
    tasks.innerText = 'Tasks';
    tasks.id = 'tasks';
    var task_paragraph = document.createElement('p');
    task_paragraph.style.fontSize = '18px';
    task_paragraph.style.fontWeight = 'normal';
    task_paragraph.style.marginLeft = '20px';
    task_paragraph.style.marginTop = '0';
    task_paragraph.style.lineHeight = '1.5';
    var bboxElements = document.getElementsByClassName('bbox');
    var taskCounter = 0;
    for (var i = 0; i < bboxElements.length; i++) {
        var h3Elements = bboxElements[i].getElementsByTagName('h3');

        for (var j = 0; j < h3Elements.length; j++) {
            var taskItem = document.createElement('a');
            var beforeContent = window.getComputedStyle(h3Elements[j], '::before').getPropertyValue('content');
            beforeContent = beforeContent.replace(/^"(.*)"$/, '$1'); // Remove quotes
            taskItem.href = '#' + h3Elements[j].id;
            if (beforeContent.match("Task") != null) {
                taskCounter++;
                var idNumber = h3Elements[j].id.replaceAll("_", " ");
                taskItem.innerText = "Task" + ' ' + taskCounter + ": " + idNumber; // Use "Task" as a string
                task_paragraph.appendChild(taskItem);
                task_paragraph.appendChild(document.createElement('br'));
            }
        }
    }
    toc.appendChild(tasks);
    tasks.appendChild(task_paragraph);

    var overviewElement = document.getElementById('overview');
    overviewElement.parentNode.insertBefore(toc, overviewElement.nextSibling);

}
