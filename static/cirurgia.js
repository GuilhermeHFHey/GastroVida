
$(document).ready(function(){
    $('#data').mask('00/00/0000');
    $("#form").trigger('reset');
});

var key = CryptoJS.enc.Utf8.parse('1234567890123456'); // TODO change to something with more entropy

function encrypt(msgString, key) {
    // msgString is expected to be Utf8 encoded
    var iv = CryptoJS.lib.WordArray.random(16);
    var encrypted = CryptoJS.AES.encrypt(msgString, key, {
        iv: iv
    });
    return iv.concat(encrypted.ciphertext).toString(CryptoJS.enc.Base64);
}

function decrypt(ciphertextStr, key) {
    var ciphertext = CryptoJS.enc.Base64.parse(ciphertextStr);

    // split IV and ciphertext
    var iv = ciphertext.clone();
    iv.sigBytes = 16;
    iv.clamp();
    ciphertext.words.splice(0, 4); // delete 4 words = 16 bytes
    ciphertext.sigBytes -= 16;
    
    // decryption
    var decrypted = CryptoJS.AES.decrypt({ciphertext: ciphertext}, key, {
        iv: iv
    });
    return decrypted.toString(CryptoJS.enc.Utf8);
}

function criptografarChaveSimetrica(data) {

    // converte para JSON
    var valores = JSON.stringify(data);
    var iv = CryptoJS.lib.WordArray.random(16);
    var encrypted = CryptoJS.AES.encrypt(valores, key, {
        iv: iv
    });
    return iv.concat(encrypted.ciphertext).toString(CryptoJS.enc.Base64);
}

toastr.options = {
  "closeButton": true,
  "debug": false,
  "newestOnTop": false,
  "progressBar": false,
  "positionClass": "toast-top-right",
  "preventDuplicates": false,
  "onclick": null,
  "showDuration": "300",
  "hideDuration": "1000",
  "timeOut": "5000",
  "extendedTimeOut": "1000",
  "showEasing": "swing",
  "hideEasing": "linear",
  "showMethod": "fadeIn",
  "hideMethod": "fadeOut"
}

$(document).on('submit', '#form',function(e){
    e.preventDefault();


    var data = {"cx": $('#cx').val(), "data": $('#data').val(),"alta": $('#alta').val(), 'pesoPreOp': $('#pesoPreOp').val()};
    

    var dataCripto = criptografarChaveSimetrica(data)

    $.ajax({
        type:'POST',
        url: $(this).attr('action'),
        data:{dados: dataCripto,
            csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val(),
            action: 'post'
        },
        success: function (response) {
            toastr["success"]("Cirurgia Registrada", "Sucesso",
            { onHidden: function() {
               window.location = window.location.protocol + "//" + window.location.host + "/view/" + pk + "/";
           }})
        },
        error: function (response) {
            toastr["error"]("Algo Falhou", "Falha")
            $("#form").trigger('reset');
        }
    });
});