(function(win,doc){
    'use strict';



    const user_input = $("#search")
    const artists_div = $('#replaceable-content')
    const endpoint = ''
    const delay_by_in_ms = 700
    let scheduled_function = false

    let ajax_call = function (endpoint, request_parameters) {
        $.getJSON(endpoint, request_parameters)
            .done(response => {
                // fade out the artists_div, then:
                artists_div.fadeTo('slow', 0).promise().then(() => {
                    // replace the HTML contents
                    artists_div.html(response['html_from_view'])
                    // fade-in the div with new contents
                    artists_div.fadeTo('slow', 1)
                })
            })
    }

    user_input.on('keyup', function () {

        const request_parameters = {
            q: $(this).val() // value of user_input: the HTML element with ID user-input
        }
    
    
        // if scheduled_function is NOT false, cancel the execution of the function
        if (scheduled_function) {
            clearTimeout(scheduled_function)
        }
    
        // setTimeout returns the ID of the function to be executed
        scheduled_function = setTimeout(ajax_call, delay_by_in_ms, endpoint, request_parameters)
    })


    //Ajax do form
    if(doc.querySelector('#form')){
        let form=doc.querySelector('#form');
        function sendForm(event)
        {
            event.preventDefault();
            let data = new FormData(form);
            let ajax = new XMLHttpRequest();
            let token = doc.querySelectorAll('input');
            ajax.open('POST', form.action);
            ajax.setRequestHeader('X-CSRF-TOKEN', token);
            ajax.onreadystatechange = function()
            {
                if(ajax.status === 200 && ajax.readyState === 4){
                    let result = doc.querySelector('#result');
                    result.innerHTML = 'Operação Realizada com Sucesso!';
                    result.classList.add('alert');
                    result.classList.add('alert-success');
                }
            }
            ajax.send(data);
            form.reset();
        }
        form.addEventListener('submit',sendForm,false);
    }

})(window,document)